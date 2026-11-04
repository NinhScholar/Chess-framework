"""Reference-line scoring and isolated-agent lifecycle for mating puzzles."""

import hashlib
import inspect
import random
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import chess

from .puzzles_999 import board_for


def _plain_stats(agent):
    stats = getattr(agent, "last_stats", {})
    if not isinstance(stats, dict):
        return {}
    return {str(key): value for key, value in stats.items()
            if value is None or type(value) in (int, float, str, bool)}


def run_puzzle(agent, puzzle, time_limit=None):
    """Recorded defensive replies; deviations are not adjudicated as losses.

    Any immediate solver checkmate is accepted, including alternative mating
    squares. Equality puzzles also accept an immediate terminal draw. Otherwise
    the recorded solver moves must match. A full equality-line match is not a
    completed game draw unless the board actually has a terminal draw outcome.
    """
    board = board_for(puzzle)
    color = board.turn
    result = {"first_move_match": None, "solution_match": False,
              "elapsed_s": 0.0, "moves": []}

    def finish(status, error=None, accepted_reason=None):
        outcome = board.outcome(claim_draw=True)
        result.update(status=status, solution_match=status == "passed", error=error,
                      accepted_reason=accepted_reason, final_fen=board.fen(),
                      terminal_result=outcome.result() if outcome else None,
                      terminal_reason=outcome.termination.name if outcome else None)
        return result

    if getattr(agent, "interactive", False):
        return finish("agent_error", "Interactive agents cannot solve headless puzzles")
    for index, expected_uci in enumerate(puzzle["solution_uci"]):
        expected = chess.Move.from_uci(expected_uci)
        if index % 2:
            board.push(expected)  # Legality was checked when loading the dataset.
            continue
        private = board.copy(stack=True)
        before = private.fen(), tuple(private.move_stack)
        started = time.perf_counter()
        try:
            move = (agent.choose_move(private) if time_limit is None
                    else agent.choose_move(private, time_limit=time_limit))
        except Exception as exc:
            result["elapsed_s"] += time.perf_counter() - started
            return finish("agent_error", f"{type(exc).__name__}: {exc}")
        elapsed = time.perf_counter() - started
        result["elapsed_s"] += elapsed
        entry = {"solver_move": index // 2 + 1, "expected": expected_uci,
                 "chosen": move.uci() if isinstance(move, chess.Move) else str(move),
                 "elapsed_s": elapsed, "stats": _plain_stats(agent)}
        result["moves"].append(entry)
        if before != (private.fen(), tuple(private.move_stack)):
            return finish("input_mutation", "Agent modified its supplied board/history")
        if not isinstance(move, chess.Move) or move not in board.legal_moves:
            return finish("illegal_move", f"Agent returned an illegal move: {move}")
        entry["chosen_san"] = board.san(move)
        matched = move == expected
        board.push(move)
        outcome = board.outcome(claim_draw=True)
        immediate_mate = board.is_checkmate() and outcome.winner == color
        terminal_draw = (puzzle["category"] == "equality_defense"
                         and outcome is not None and outcome.winner is None)
        accepted = matched or immediate_mate or terminal_draw
        entry["accepted"] = accepted
        if index == 0:
            result["first_move_match"] = accepted
        if immediate_mate or terminal_draw:
            return finish("passed", accepted_reason="checkmate" if immediate_mate else "terminal_draw")
        if not matched:
            return finish("reference_deviation")
    return finish("passed", accepted_reason="reference_line")


def _worker(connection, agent_name, puzzle, seed, time_limit, agent_options=None):
    agent = None
    result = None
    try:
        from agents import AGENT_FACTORIES

        random.seed(seed)
        factory = AGENT_FACTORIES[agent_name]
        constructor_seed = "seed" in inspect.signature(factory).parameters
        options = dict(agent_options or {})
        if constructor_seed:
            options["seed"] = seed
        agent = factory(**options)
        getattr(agent, "on_game_start", lambda color: None)(board_for(puzzle).turn)
        result = run_puzzle(agent, puzzle, time_limit)
        result["seed_method"] = "constructor_and_python_random" if constructor_seed else "python_random"
        result["agent_class"] = f"{type(agent).__module__}.{type(agent).__qualname__}"
        result["agent_config"] = {
            key: getattr(agent, key) for key in (
                "simulations", "rollout_depth", "exploration", "seed",
                "rollout_randomness", "progressive_expansion", "widening_base",
                "widening_exponent", "exchange_safety")
            if hasattr(agent, key)
        }
        source = inspect.getsourcefile(type(agent))
        result["agent_source_sha256"] = (hashlib.sha256(Path(source).read_bytes()).hexdigest()
                                           if source else None)
        if result["terminal_result"] is not None:
            getattr(agent, "on_game_end", lambda outcome: None)(result["terminal_result"])
    except BaseException as exc:
        result = {"status": "agent_error", "first_move_match": None,
                  "solution_match": False, "elapsed_s": None, "moves": [],
                  "error": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc(limit=5)}
    finally:
        if agent is not None:
            try:
                getattr(agent, "close", lambda: None)()
            except Exception as exc:
                if result is not None:
                    result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        try:
            connection.send(result)
        finally:
            connection.close()


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record["agent"], "all")].append(record)
        groups[(record["agent"], record["category"])].append(record)
    result = {}
    for (agent, category), rows in groups.items():
        elapsed = [row["elapsed_s"] for row in rows if row.get("elapsed_s") is not None]
        counts = dict(Counter(row["status"] for row in rows))
        result.setdefault(agent, {})[category] = {
            "trials": len(rows), "first_move_matches": sum(bool(r["first_move_match"]) for r in rows),
            "first_move_match_rate": sum(bool(r["first_move_match"]) for r in rows) / len(rows),
            "solution_matches": sum(bool(r["solution_match"]) for r in rows),
            "solution_match_rate": sum(bool(r["solution_match"]) for r in rows) / len(rows),
            "status_counts": counts,
            "mean_search_s": sum(elapsed) / len(elapsed) if elapsed else None,
            "mean_wall_s": sum(row["wall_s"] for row in rows) / len(rows),
        }
    return result
