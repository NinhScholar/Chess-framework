"""Run registered agents or module:factory agents on the supplied mating puzzles."""

import argparse
import concurrent.futures
import importlib
import inspect
import json
import math
import multiprocessing
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import reference
from .puzzles_999 import DEFAULT_DATASET, board_for, load_dataset


def get_factory(name):
    from agents import AGENT_FACTORIES
    from agents.mcts_agent import MCTSAgent

    if name == "MCTS":
        # The game's zero-argument registry lambda cannot accept a search budget.
        return MCTSAgent
    if name in AGENT_FACTORIES:
        return AGENT_FACTORIES[name]
    if ":" in name:
        module, attribute = name.split(":", 1)
        factory = getattr(importlib.import_module(module), attribute)
        if callable(factory):
            return factory
    raise ValueError(f"Unknown agent {name!r}; use --list-agents or module:Class")


def accepts(factory, parameter):
    signature = inspect.signature(factory)
    return parameter in signature.parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())


def agent_options(name, args, supplied):
    factory = get_factory(name)
    if getattr(factory, "interactive", False):
        raise ValueError(f"{name}: interactive agents cannot solve headless puzzles")
    options = dict(supplied)
    if "seed" in options or "simulations" in options:
        raise ValueError("Use --seeds and --simulations instead of JSON seed/simulations")
    if accepts(factory, "simulations"):
        options["simulations"] = args.simulations
    for field in ("rollout_depth", "exploration"):
        value = getattr(args, field)
        if value is not None:
            if not accepts(factory, field):
                raise ValueError(f"{name} does not accept --{field.replace('_', '-')}")
            options[field] = value
    for field in ("rollout_depth",):
        if field in options and (type(options[field]) is not int or options[field] < 0):
            raise ValueError(f"{name}: {field} must be a nonnegative integer")
    if "exploration" in options and (not isinstance(options["exploration"], (int, float))
                                    or not math.isfinite(options["exploration"])
                                    or options["exploration"] < 0):
        raise ValueError(f"{name}: exploration must be finite and nonnegative")
    bound = dict(options)
    if "seed" in inspect.signature(factory).parameters:
        bound["seed"] = args.seeds[0]
    inspect.signature(factory).bind(**bound)
    return options


def _worker(connection, name, puzzle, seed, options):
    """Adapt only this fresh worker; reuse the existing scorer and lifecycle."""
    from agents import AGENT_FACTORIES

    try:
        factory = get_factory(name)
    except BaseException as exc:
        connection.send({"status": "agent_error", "error": f"{type(exc).__name__}: {exc}"})
        connection.close()
        return
    missing = object()
    previous_factory = AGENT_FACTORIES.get(name, missing)
    previous_loader = reference.board_for
    try:
        AGENT_FACTORIES[name] = factory
        reference.board_for = board_for
        reference._worker(connection, name, puzzle, seed, None, options)
    finally:
        reference.board_for = previous_loader
        if previous_factory is missing:
            AGENT_FACTORIES.pop(name, None)
        else:
            AGENT_FACTORIES[name] = previous_factory


def run_trial(name, puzzle, seed, options, timeout):
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(send, name, puzzle, seed, options))
    started = time.perf_counter()
    try:
        process.start()
        send.close()
        if receive.poll(max(0.0, timeout - (time.perf_counter() - started))):
            try:
                result = receive.recv()
            except EOFError:
                result = {"status": "worker_error", "error": "Worker exited without a result"}
        else:
            result = {"status": "timeout", "error": f"Trial exceeded {timeout:g} seconds"}
    except Exception as exc:
        result = {"status": "worker_error", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        send.close()
        receive.close()
        if process.pid is not None:
            process.join(timeout=0.2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
                if process.is_alive():
                    process.kill()
                    process.join()
            process.close()
    if not isinstance(result, dict):
        result = {"status": "worker_error", "error": "Invalid worker result"}
    for key, value in {"first_move_match": None, "solution_match": False,
                       "elapsed_s": None, "moves": []}.items():
        result.setdefault(key, value)
    result.update(agent=name, puzzle_id=puzzle["id"], source_index=puzzle["source_index"],
                  category=puzzle["category"], seed=seed, agent_options=options,
                  wall_s=time.perf_counter() - started)
    return result


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_report(path, summary):
    lines = ["# Puzzle benchmark", "", "Scores are completed reference solutions or immediate solver mates, not game wins.", "",
             "| Agent | Category | Trials | Solved | First moves | Mean search seconds |",
             "|---|---|---:|---:|---:|---:|"]
    for name, groups in summary.items():
        for category, values in groups.items():
            elapsed = values["mean_search_s"]
            time_text = f"{elapsed:.3f}" if elapsed is not None else "n/a"
            lines.append(f"| {name} | {category} | {values['trials']} | "
                         f"{values['solution_matches']} | {values['first_move_matches']} | {time_text} |")
    for name, groups in summary.items():
        lines.extend(["", f"{name} status counts: {groups['all']['status_counts']}", ""])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser_for_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", nargs="+", default=["mcts_improve"])
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--timeout", type=float, default=60.0, help="Hard seconds per entire puzzle trial")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, help="Smoke test only this many selected puzzles")
    parser.add_argument("--category", nargs="+")
    parser.add_argument("--puzzle-ids", nargs="+")
    parser.add_argument("--rollout-depth", type=int)
    parser.add_argument("--exploration", type=float)
    extras = parser.add_mutually_exclusive_group()
    extras.add_argument("--options-file", type=Path, help="JSON object mapping agent names to constructor options")
    extras.add_argument("--agent-options", help="Inline equivalent of --options-file")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--list-agents", action="store_true")
    return parser


def main(argv=None):
    parser = parser_for_cli()
    args = parser.parse_args(argv)
    if args.list_agents:
        from agents import AGENT_FACTORIES
        print("\n".join(AGENT_FACTORIES))
        print("Custom agents: module:Class (for example agents.my_agent:MyAgent)")
        return 0
    if args.simulations <= 0 or args.workers <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("simulations, workers and limit must be positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    if len(set(args.agent)) != len(args.agent) or len(set(args.seeds)) != len(args.seeds):
        parser.error("agents and seeds must not repeat")
    try:
        supplied = json.loads(args.options_file.read_text(encoding="utf-8-sig")
                              if args.options_file else args.agent_options or "{}")
        if not isinstance(supplied, dict) or set(supplied) - set(args.agent):
            raise ValueError("options must map selected agent names to constructor options")
        if any(not isinstance(value, dict) for value in supplied.values()):
            raise ValueError("each agent's options must be a JSON object")
        options = {name: agent_options(name, args, supplied.get(name, {})) for name in args.agent}
        payload = load_dataset(args.dataset)
    except (OSError, ValueError, TypeError, ImportError, AttributeError) as exc:
        parser.error(str(exc))
    if args.validate_only:
        print(json.dumps(payload["metadata"], indent=2))
        return 0
    puzzles = payload["puzzles"]
    categories = {p["category"] for p in puzzles}
    if args.category:
        if set(args.category) - categories:
            parser.error("unknown category; use --validate-only to list categories")
        puzzles = [p for p in puzzles if p["category"] in args.category]
    if args.puzzle_ids:
        if set(args.puzzle_ids) - {p["id"] for p in puzzles}:
            parser.error("unknown puzzle ID or ID outside selected categories")
        puzzles = [p for p in puzzles if p["id"] in args.puzzle_ids]
    if args.limit:
        puzzles = puzzles[:args.limit]
    if not puzzles:
        parser.error("no puzzles selected")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or Path("benchmark_results") / ("999-" + stamp)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        parser.error(f"Cannot create a new output directory: {exc}")
    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "complete": False,
        "dataset": payload["metadata"], "selected_puzzle_count": len(puzzles),
        "agents": args.agent, "seeds": args.seeds, "agent_options": options,
        "puzzle_ids": [p["id"] for p in puzzles], "workers": args.workers,
        "hard_trial_timeout_s": args.timeout, "time_limit_s": None,
        "python": sys.version,
        "scoring": "Recorded opponent replies; reference solver moves or immediate solver checkmate accepted. Other legal deviations are unadjudicated puzzle mismatches, not game losses.",
    }
    write_json(output / "puzzles.json", {**payload, "puzzles": puzzles,
                                        "selected_puzzle_count": len(puzzles)})
    records = []

    def checkpoint():
        write_json(output / "summary.json", {"metadata": metadata, "summary": reference.summarize(records)})

    checkpoint()
    total = len(puzzles) * len(args.agent) * len(args.seeds)
    print(f"Validated {payload['metadata']['puzzle_count']} puzzles; running {total} trials.", flush=True)
    print(f"Output: {output.resolve()}", flush=True)
    print(f"Constructor options: {json.dumps(options)}", flush=True)
    try:
        with (output / "results.jsonl").open("w", encoding="utf-8") as stream:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                try:
                    futures = {
                        pool.submit(run_trial, name, puzzle, seed, options[name], args.timeout): (name, puzzle, seed)
                        for seed in args.seeds for puzzle in puzzles for name in args.agent
                    }
                    for future in concurrent.futures.as_completed(futures):
                        name, puzzle, seed = futures[future]
                        try:
                            record = future.result()
                        except Exception as exc:
                            record = {"agent": name, "puzzle_id": puzzle["id"], "category": puzzle["category"],
                                      "seed": seed, "status": "worker_error", "error": str(exc),
                                      "solution_match": False, "first_move_match": None,
                                      "elapsed_s": None, "wall_s": 0.0, "moves": []}
                        records.append(record)
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stream.flush()
                        print(f"[{len(records)}/{total}] {name} seed={seed} {puzzle['id']}: {record['status']}", flush=True)
                        if len(records) % 10 == 0:
                            checkpoint()
                except KeyboardInterrupt:
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise
    except KeyboardInterrupt:
        metadata["interrupted"] = True
        print("Interrupted. Completed trials have been saved; this run is incomplete.")
        return 130
    finally:
        checkpoint()
    metadata.update(complete=True, finished_at_utc=datetime.now(timezone.utc).isoformat())
    checkpoint()
    summary = reference.summarize(records)
    write_report(output / "summary.md", summary)
    for name in args.agent:
        result = summary[name]["all"]
        print(f"{name}: {result['solution_matches']}/{result['trials']} full solutions; "
              f"{result['first_move_matches']} accepted first moves; {result['status_counts']}")
    return int(any(r["status"] not in ("passed", "reference_deviation") for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
