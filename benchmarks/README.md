# 999-puzzle agent benchmark

## Recorded results: original versus improved MCTS

**Revision note:** these are the latest saved full runs before exchange safety
was added. They do **not** measure the current safety-enabled `mcts_improve`.
The improved run already had tactical cutoff removed and used 30 rollout plies.

Both agents used the same 999 puzzles, seed 0, 200 simulations per search,
rollout depth 30 and exploration 1.4. Each run used four isolated workers and
a 60-second hard timeout per puzzle. The runs were executed separately on
4 November 2026. Improved MCTS used progressive expansion.

| Puzzle category | Puzzles | Original MCTS solved | Improved MCTS solved |
|---|---:|---:|---:|
| Mate in one | 333 | 333 (100.00%) | 333 (100.00%) |
| Mate in two | 333 | 35 (10.51%) | 333 (100.00%) |
| Mate in three | 333 | 4 (1.20%) | 208 (62.46%) |
| **Total** | 999 | 372 (37.24%) | 874 (87.49%) |

| Additional measure | Original MCTS | Improved MCTS |
|---|---:|---:|
| Accepted first moves | 381/999 | 874/999 |
| Reference deviations | 627 | 125 |
| Execution errors / timeouts | 0 | 0 |
| Mean recorded search seconds per puzzle | 6.119 | 1.278 |

Improved MCTS completed **502 more solutions**, a gain of
**50.25 percentage points**. Both solved every mate-in-one puzzle;
the main gains were in mate-in-two and mate-in-three. This supports better
multi-move puzzle solving for that tested revision on this fixed dataset.

The timing values describe separate runs with concurrent workers; they are
not a controlled speed comparison. A matching simulation cap does not match
computation time. One seed and reference-line scoring do not establish Elo,
game win rates, or success against every possible defensive reply.

[Verified counts, configurations, dates and source fingerprints](results/999_comparison.json)
are included so the historical comparison can be checked. The latest agent
adds exchange safety; run the complete command below to measure that revision.

[Improved MCTS concepts and pseudocode](../docs/mcts_improve.md).

## Run the benchmark

Run from the `Chess-IntroAI` repository root with Python 3.10 or newer:

```powershell
python3.12 -m pip install -r requirements.txt

# Validate every puzzle and reference solution.
python3.12 test_agent_999.py --validate-only

# Compare agents on all 999 puzzles.
python3.12 test_agent_999.py --agent MCTS mcts_improve --simulations 200

# Test the random agent.
python3.12 test_agent_999.py --agent Random

# Quick check before a complete run.
python3.12 test_agent_999.py --agent MCTS mcts_improve --limit 3

# Test a future agent without changing the game registry.
python3.12 test_agent_999.py --agent agents.my_agent:MyAgent --simulations 200
```

[Detailed runner guide](README_999.md). The runner also works as
`python3.12 -m benchmarks.run_999`.

The dataset is [data/puzzles_999.json](data/puzzles_999.json), containing 333
mate-in-one, 333 mate-in-two and 333 mate-in-three puzzles. Every FEN, PGN
history, SAN/UCI solution and final checkmate is validated before running.

The agent plays against recorded opponent replies. Reference solutions and
immediate solver mates pass. A different legal move is `reference_deviation`,
not a game loss. Timeouts, illegal moves and input mutation are reported
separately. Results, settings and source hashes are saved under ignored
`benchmark_results/`, or a new directory selected with `--output`.

## Data source

The puzzles come from
[quantum24/chess_puzzles_1k_in_pgn_san](https://huggingface.co/datasets/quantum24/chess_puzzles_1k_in_pgn_san),
published by **quantum24** on Hugging Face. The dataset card lists **999 training
rows** and the **CC0-1.0** license.

The benchmark uses the local copy [data/puzzles_999.json](data/puzzles_999.json).
It was preserved byte-for-byte from the supplied development file
`tests/1000_puzzles_in_pgn_san.json`; despite that filename, it contains 999 records.
SHA256: `c963dbec96a9622262a92d865850e04d9e1787b2295d9e726f532bcb8fed541f`.
