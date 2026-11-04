"""Validate the supplied FEN/PGN/UCI/SAN mating-puzzle collection."""

import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import chess
import chess.pgn


DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "puzzles_999.json"
MATE_NAMES = {1: "mate_in_one", 2: "mate_in_two", 3: "mate_in_three"}


def board_for(puzzle):
    """Replay a coherent history, including its initial move-number counter."""
    board = chess.Board(puzzle["initial_fen"])
    for uci in puzzle["history_uci"]:
        board.push_uci(uci)
    if board.fen() != puzzle["fen"]:
        raise ValueError(f"{puzzle['id']}: history does not reproduce FEN")
    return board


def normalize_puzzle(record, index):
    for field in ("FEN", "PGN", "Moves", "MovesSAN", "Themes"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"entry {index}: missing or empty {field}")
    target = chess.Board(record["FEN"])
    if not target.is_valid():
        raise ValueError(f"entry {index}: invalid FEN")
    game = chess.pgn.read_game(io.StringIO(record["PGN"]))
    if game is None or game.errors:
        raise ValueError(f"entry {index}: invalid PGN")
    generated = game.end().board()
    if generated.fen().split()[:5] != target.fen().split()[:5]:
        raise ValueError(f"entry {index}: PGN position differs from FEN")
    initial = game.board()
    adjustment = target.fullmove_number - generated.fullmove_number
    # Correct before replay: changing only the root counter leaves stack states
    # inconsistent when python-chess checks repetition by popping/replaying moves.
    initial.fullmove_number += adjustment
    if initial.fullmove_number < 1:
        raise ValueError(f"entry {index}: incompatible PGN/FEN move counters")
    solution, sans = record["Moves"].split(), record["MovesSAN"].split()
    depth = (len(solution) + 1) // 2
    themes = record["Themes"].split()
    if (not solution or len(solution) % 2 != 1 or len(solution) != len(sans)
            or depth not in MATE_NAMES or f"mateIn{depth}" not in themes):
        raise ValueError(f"entry {index}: solution length, SAN or mate theme mismatch")
    puzzle = {
        "id": f"p{index:04d}", "source_index": index,
        "category": MATE_NAMES[depth], "initial_fen": initial.fen(),
        "history_uci": [move.uci() for move in game.mainline_moves()],
        "fen": target.fen(), "solution_uci": solution, "solution_san": sans,
        "themes": themes, "fullmove_counter_adjustment": adjustment,
        "source_url": None,
    }
    board = board_for(puzzle)
    solver = board.turn
    for uci, san in zip(solution, sans):
        if board.outcome(claim_draw=True) is not None:
            raise ValueError(f"entry {index}: terminal position before solution ends")
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves or board.parse_san(san) != move:
            raise ValueError(f"entry {index}: illegal or inconsistent solution move {uci}")
        board.push(move)
    if not board.is_checkmate() or board.outcome().winner != solver:
        raise ValueError(f"entry {index}: solution does not deliver solver checkmate")
    return puzzle


def load_dataset(path=DEFAULT_DATASET):
    path = Path(path)
    raw = path.read_bytes()
    records = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(records, list) or not records:
        raise ValueError("dataset must be a nonempty list of puzzle objects")
    puzzles = []
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"entry {index}: expected a puzzle object")
        try:
            puzzles.append(normalize_puzzle(record, index))
        except ValueError as exc:
            raise ValueError(f"entry {index}: {exc}") from exc
    positions = Counter(" ".join(p["fen"].split()[:4]) for p in puzzles)
    metadata = {
        "source": str(path.resolve()), "source_sha256": hashlib.sha256(raw).hexdigest(),
        "puzzle_count": len(puzzles),
        "categories": dict(Counter(p["category"] for p in puzzles)),
        "duplicate_positions": sum(count - 1 for count in positions.values()),
        "counter_corrections": [
            {"puzzle_id": p["id"], "source_index": p["source_index"],
             "fullmove_counter_adjustment": p["fullmove_counter_adjustment"]}
            for p in puzzles if p["fullmove_counter_adjustment"]
        ],
    }
    return {"schema_version": 1, "metadata": metadata, "puzzles": puzzles}
