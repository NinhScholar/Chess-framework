# Test an agent on the 999-puzzle set

Run these commands from the Chess-IntroAI folder with the project Python
environment active. This uses the existing project dependencies.

```powershell
# All 999 puzzles, improved MCTS, 200 simulations per search.
python test_agent_999.py --agent mcts_improve --simulations 200

# Compare the original and improved agents in one run.
python test_agent_999.py --agent MCTS mcts_improve --simulations 200

# A quick check before a long run.
python test_agent_999.py --agent MCTS mcts_improve --limit 3

# Validate all input records without running an agent.
python test_agent_999.py --validate-only

# Test a future agent without adding it to the game's registry.
python test_agent_999.py --agent agents.my_agent:MyAgent --simulations 200
```

Dataset source:
[quantum24/chess_puzzles_1k_in_pgn_san on Hugging Face](https://huggingface.co/datasets/quantum24/chess_puzzles_1k_in_pgn_san).
Its dataset card lists 999 training rows and the CC0-1.0 license. See
[Data source](README.md#data-source) for the local copy and its fingerprint.

The current file `benchmarks/data/puzzles_999.json` contains **999** puzzles:
333 each of mate-in-one, mate-in-two and mate-in-three. Every input position,
PGN history, SAN/UCI solution and final checkmate is validated before testing.
The three PGN/FEN move-number differences are reconciled before history replay,
preserving consistent counters through repetition checks. The source data is
never edited.

The default agent is `mcts_improve`, with 200 simulations and its normal
constructor settings: rollout depth 30, exploration 1.4 and progressive
expansion and exchange safety enabled. `MCTS` also uses rollout depth 30. The simulation budget is
a constructor option; the agents' own simulation defaults stay unchanged.
Simulation budgets are passed to factories
that accept `simulations`; other factories keep their own budget conventions.
The actual configuration and source hash are recorded in every normal trial.

Useful options:

- `--rollout-depth 4 --exploration 1.4` overrides supported
  constructor parameters. Explicit CLI options take priority over JSON options.
- `--seeds 0 1 2`, `--workers 4`, and `--timeout 60` control repetitions,
  concurrent isolated trials and the hard timeout for an entire puzzle.
- `--category mate_in_three` or `--puzzle-ids p0565 p0903 p0911` selects positions.
  IDs correspond to one-based file entries. The complete input is still validated.
- `--output path/to/new-folder` chooses a new output directory. Existing folders
  are never overwritten; the default is `benchmark_results/999-<UTC timestamp>`.
- `--list-agents` lists registered names. A custom `module:Class` must be importable
  from this environment and follow the existing `choose_move` agent interface.

For additional constructor parameters, put a JSON object such as the following
in `options.json`, then add `--options-file options.json`:

```json
{"mcts_improve": {"widening_base": 3.0, "rollout_randomness": 0.1}}
```

Use the CLI flags for simulations and seeds rather than putting those in JSON.
Each trial has a fresh process and agent. Results are flushed to `results.jsonl`
after each puzzle, and `summary.json` is updated throughout the run. The final
`summary.md` reports full-solution scores by category, first-move scores, search
times and error counts. `puzzles.json` preserves the normalized positions and
available histories used in that run.

Ctrl+C cancels queued trials and retains completed records with an incomplete
summary. Trials already running finish or reach their hard timeout.

The agent plays the solver's turns against the supplied opponent replies. A
reference solution or any immediate solver checkmate is accepted. Other legal
deviations are puzzle mismatches, not game losses. These are reference-line tests,
not exhaustive proofs against every possible defense or game win-rate tests.
Crashes, illegal moves, board mutation and timeouts are reported separately.

See the [recorded agent comparison](README.md#recorded-results-original-versus-improved-mcts)
and the [improved MCTS guide](../docs/mcts_improve.md).
