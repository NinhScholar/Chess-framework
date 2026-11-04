# mcts_improve: concepts, implementation and pseudocode

`MCTSImproveAgent` is a chess agent built around Monte Carlo Tree Search (MCTS).
It adds exchange-aware move safety, progressive expansion, heuristic leaf evaluation
and exact outcome propagation. It uses `python-chess` for chess rules and does
not call Stockfish or a neural network.

This guide describes [the current implementation](../agents/mcts_improve.py). The original
agent remains available as `MCTS`; the improved agent is registered as
`mcts_improve`. All pseudocode below is explanatory and uses the implementation's
root-player value convention.

## Overall idea

The improved agent keeps the MCTS cycle: select a branch, expand a move,
simulate a continuation, and back up its value. It uses chess-specific hints
to spend a limited simulation budget more carefully while keeping estimates
separate from exact results.

1. **Choose where to search.** Opponent-aware UCT balances promising actions
   with less-visited alternatives. Progressive expansion limits width below
   the root so some search effort can go deeper.
2. **Try useful moves early.** Estimate material gained after the opponent's
   best legal capture exchange. Order safer moves early, guide rollouts away
   from losing material, and use safety to break equal visit/value ties.
   Checks, castling and centralization still contribute to move ordering.
3. **Score unfinished continuations.** A bounded material and positional
   evaluation scores the board when the rollout reaches its ply limit.
4. **Trust exact evidence separately.** Terminal rules, immediate mates,
   forced replies and solved children can establish exact results. A sampled
   high score cannot create a proof. Final selection protects a proven draw
   unless a win has been proved.
5. **Keep search practical.** Nodes store moves and statistics; one private
   board is pushed and popped while preserving the supplied history. Local
   seeding, deadline checks and counters make searches easier to reproduce
   and inspect.

The overall idea is therefore **informed MCTS with reliable outcome handling**:
use sampling to allocate effort, chess heuristics to guide that sampling, and
exact checks to recognize results that no longer need guessing. The agent
starts a new tree for each move; it does not train a model between puzzles.

## Overall pseudocode

This overview shows the complete search flow. Helper details and exact budget
handling are expanded in [Search and proof pseudocode](#4-search-and-proof-pseudocode).

```text
FUNCTION MCTS_IMPROVED(board, simulation_limit, rollout_depth):

    IF board is terminal:
        RETURN None

    IF board has only one legal move:
        RETURN that move

    root_player ← board.turn
    working_board ← copy board, preserving history
    fallback ← legal move with smallest UCI notation
    TRY:
        root ← create and prepare root node
    ON deadline expiration:
        RETURN fallback

    IF root has an immediate mating move:
        RETURN that move

    WHILE simulation budget remains AND root is unresolved:
        node ← root
        path ← [root]

        TRY:
            # 1. Selection and expansion
            WHILE node is unresolved:
                IF expansion is allowed:
                    move ← highest-priority untried move
                    safety ← cached MOVE_SAFETY(working_board, move)
                    play move on working_board
                    node ← create and prepare child with incoming safety
                    append node to path
                    BREAK
                ELSE:
                    node ← select child using opponent-aware UCT
                    play node.move on working_board
                    append node to path

            # 2. Evaluation
            IF node has a proven result:
                reward ← proven result
            ELSE:
                simulate guided moves up to rollout_depth
                IF resulting position is terminal:
                    reward ← terminal result
                ELSE:
                    reward ← static material/position evaluation

            check deadline

        ON deadline expiration:
            discard incomplete iteration
            BREAK

        FINALLY:
            restore working_board to the root position

        # 3. Backup
        FOR each node in path, from leaf to root:
            node.visits ← node.visits + 1
            node.value ← node.value + reward
            update node's proven result from solved children

    # 4. Final decision
    prioritize:
        proven win → proven draw → unresolved move → proven loss

    within that group:
        choose most visits, then highest value, then best material safety,
        then UCI tie-break

    RETURN chosen move, or fallback
```

- **Prepare:** check terminal outcomes, immediate mates and short forced
  continuations; order remaining moves.
- **Expansion:** cover root actions when the budget permits, then gradually
  widen deeper nodes.
- **Rewards:** always use the root player's perspective: win `1`, draw `0.5`,
  loss `0`.
- **Proofs:** heuristic estimates never establish a proven result.

Contents:

- [Overall idea](#overall-idea)
- [Overall pseudocode](#overall-pseudocode)

1. [MCTS fundamentals](#1-mcts-fundamentals)
2. [Changes from the original agent](#2-changes-from-the-original-agent)
3. [Nodes, rewards and UCT](#3-nodes-rewards-and-uct)
4. [Search and proof pseudocode](#4-search-and-proof-pseudocode)
5. [Rollouts and static evaluation](#5-rollouts-and-static-evaluation)
6. [Parameters and budgets](#6-parameters-and-budgets)
7. [Running the agent](#7-running-the-agent)
8. [Measured evaluation](#8-measured-evaluation)
9. [Limitations](#9-limitations)
10. [Code map and further reading](#10-code-map-and-further-reading)

## 1. MCTS fundamentals

A chess position can have many legal moves. Exploring every continuation to the
end of the game is impractical, so this agent builds a selective search tree.
Each iteration uses previous estimates to choose a branch, explores one new
action when permitted, evaluates its continuation, and updates the path.

| Phase | Purpose | What this implementation does |
| --- | --- | --- |
| Selection | Decide which existing branch deserves another visit | Adversarial UCT over unresolved children |
| Expansion | Add a new action to the tree | Tactical ordering and optional progressive expansion |
| Evaluation | Estimate the resulting position | Exact node result, bounded rollout, or static cutoff |
| Backup | Update knowledge along the selected path | Add the same root-player reward to every visited node; update proofs separately |

A **ply** is one move by one player. A **simulation** here means one completed
selection/expansion/evaluation/backup iteration; it does not mean a complete game.
A rollout is the sampled continuation used inside an iteration. An iteration can
instead evaluate a leaf directly, without playing any rollout moves.

**Exploration** gives attention to less-visited moves. **Exploitation** favors
moves with better observed values. UCT combines them in its selection score.
The foundational UCT paper is [Kocsis and Szepesvari, 2006](https://aima.cs.berkeley.edu/~russell/classes/cs294/s11/readings/Kocsis%2BSzepesvari%3A2006.pdf).

## 2. Changes from the original agent

Both agents already use MCTS, root-player rewards, and adversarial opponent
selection. The original also copies boards with available history. Those are
shared foundations rather than new features of the improved agent.

| Area | Original `MCTSAgent` | `MCTSImproveAgent` |
| --- | --- | --- |
| Node storage | Full board at every node | Moves, turn and statistics; one private board for traversal |
| Expansion order | Random untried move | Exchange safety plus checks, castling and center bonuses; random ties |
| Expansion width | Expand untried actions before UCT | Same at root; optional gradual width growth below root |
| Rollout policy | Uniform random legal moves | Safety-checked random/weighted moves; least-loss fallback |
| Nonterminal estimate | Clipped material difference | Smooth material and positional estimate |
| Exact tree results | Sample terminal outcomes | Separate proven terminal, mate-in-one, forced-chain and solved-child results |
| Final choice | Most visited root action | Proven-result priority, then visits, value, safety and UCI tie-break |
| Randomness | Python's global random generator | Local generator restarted from `seed` for each search |
| Budgets | A supplied time limit replaces the simulation condition | Simulation cap and optional deadline both apply |
| Inspection | No detailed search counters | `last_stats` and `last_root_stats` |

The improved agent combines sampling, informed move ordering and explicit
mate checks. Nonterminal rollout endpoints use the static evaluator directly.

## 3. Nodes, rewards and UCT

### Node contents

For a node `v`:

| Field | Meaning |
| --- | --- |
| `turn` | Player to move at this position |
| `move` | Move from the parent to this node |
| `parent`, `children` | Tree links |
| `untried` | Legal actions not yet added as children, ordered by priority |
| `visits` or `N(v)` | Number of completed iterations backed up through the node |
| `value` or `W(v)` | Sum of backed-up rewards for the search-root player |
| `mean` or `Q(v)` | `W(v) / N(v)`; 0.5 if unvisited |
| `proven` | `None` if unresolved; otherwise an exact root-player outcome |
| `mate_move` | An immediate mating move found while preparing the node |
| `safety` | Material-safety estimate for the incoming move, in its mover's perspective |

A new tree is built for every `choose_move` call. There is no tree reuse between
moves and no tree-statistics transposition table. A bounded per-search cache
reuses material-safety calculations; it does not share visits, values or proofs.

### Reward convention

Let `root_color` be the player choosing the current move:

```text
root player wins:  1.0
terminal draw:    0.5
root player loses: 0.0
```

Every node stores values from that same perspective. If Black is the root
player, a Black win is still 1.0. The value is not always White's score and is
not automatically the score of the player to move at a particular node.

During statistical backup, the reward is added unchanged at every ancestor.
The algorithm does not alternate `r` and `1-r` while backing up. Opponent intent
is represented during selection and proof propagation instead.

### Adversarial UCT

At parent `p`, select an unresolved child `v` maximizing:

```text
exploitation(v) = Q(v)       when p.turn == root_color
                  1 - Q(v)  otherwise

UCT(v) = exploitation(v)
         + exploration * sqrt(ln(max(1, N(p))) / N(v))
```

An unvisited child receives an infinite selection score. Proven children are
excluded from UCT because their exact result needs no further sampling. The
exploration bonus can make UCT exceed 1; UCT itself is not a probability.

Example: at an opponent node, children with root-player means 0.8 and 0.2 have
exploitation terms 0.2 and 0.8. With equal visit counts, the opponent prefers the
second action because it is worse for the root player.

### Estimated versus proven values

`mean=0.9, proven=None` is an optimistic estimate, not a demonstrated win.
A rollout that ends in checkmate proves only what happened along that sampled
continuation. It does not prove that all opponent alternatives also lose.

Exact tree results are stored separately. Propagating solved results through
MCTS is related to [MCTS-Solver](https://dke.maastrichtuniversity.nl/m.winands/documents/uctloa.pdf).
The rules below describe this project's implementation, rather than claiming
that it duplicates every detail of that paper.

## 4. Search and proof pseudocode

Notation: `PREPARE` checks exact results and orders legal moves.
`RESTORE_ROOT` pops the private board back to its
original stack depth. Deadline checks in the code also occur inside helpers.

### Main search

```text
CHOOSE_MOVE(input_board, optional_time_limit):
    reset search statistics
    deadline = search_start + time_limit, or no deadline if unspecified
    if input position is terminal, including a claimable draw:
        return None
    legal = legal moves
    fallback = legal move with smallest UCI notation
    if legal contains only one move:
        return that move

    B = private copy of input_board, preserving its history
    root_color = B.turn
    rng = new local generator seeded with self.seed
    root = new node(turn=B.turn)
    try:
        PREPARE(B, root, forced_depth=2)
    on deadline expiration:
        return fallback

    while completed_iterations < simulations and root.proven is unknown:
        node = root
        path = [root]
        try:
            CHECK_DEADLINE()
            while node.proven is unknown:
                CHECK_DEADLINE()
                if CAN_EXPAND(node):
                    move = highest-priority untried action
                    push move onto B
                    child = new node(turn=B.turn, move=move, parent=node)
                    PREPARE(B, child, forced_depth=2)
                    remove move from node.untried
                    attach child to node.children
                    node = child
                    append node to path
                    break
                node = unresolved child with highest adversarial UCT
                push node.move onto B
                append node to path

            if node.proven is known:
                result = node.proven
            else:
                result = EVALUATE_WITH_ROLLOUT(B)
            CHECK_DEADLINE()
        on deadline expiration:
            count discarded iteration
            break
        finally:
            RESTORE_ROOT(B)

        for v in path, from leaf to root:
            v.visits += 1
            v.value += result
            UPDATE_PROVEN(v)
        completed_iterations += 1

    if root has an immediate mating move:
        return root.mate_move
    return BEST_ROOT_ACTION(root), or fallback if none is eligible
```

The board is restored even if an iteration times out. An incomplete iteration
adds no visits or sampled value. A valid exact proof discovered before the
interruption can still influence the final move.

### Progressive expansion

```text
CAN_EXPAND(node):
    if node.untried is empty:
        return false
    if progressive_expansion is disabled or node is the root:
        return true
    if every admitted child is proven:
        return true
    allowance = max(1, floor(widening_base
                             * max(1, node.visits) ** widening_exponent))
    return number_of_children < allowance
```

With defaults, the allowance is approximately `2 * sqrt(visits)`: 4 children at
4 visits, 8 at 16 visits, and 20 at 100 visits. Below the root, search may descend
before all legal actions have been expanded. Untried actions remain available
and must still be accounted for when establishing exact results.

At the root, expansion is always permitted while untried moves remain. This
encourages root coverage before UCT selection; a proof or exhausted budget can
still stop the search before every root action is explored.

### Preparing a node and finding exact results

```text
PREPARE(B, node, forced_depth):
    CHECK_DEADLINE()
    if B has a terminal outcome under claim_draw=True:
        node.proven = terminal utility for root_color
        return
    moves = legal moves in B

    if exactly one legal move exists and forced_depth > 0:
        push that move
        try:
            temporary = new node(turn=B.turn)
            PREPARE(B, temporary, forced_depth - 1)
        finally:
            pop move
        if temporary.proven is known:
            node.proven = temporary.proven
            return

    for move in moves:
        CHECK_DEADLINE()
        if move gives check and actually delivers checkmate:
            node.proven = 1 if node.turn == root_color else 0
            node.mate_move = move
            return
    node.untried = moves sorted by tactical priority, with local random ties
```

A forced continuation means **only one legal move**, not merely a likely reply.
The helper follows at most two such moves and can detect a mate-in-one at the
resulting position. This is a small exact shortcut, not a general three-ply
minimax search over all branches.

### Exact-result backup

```text
UPDATE_PROVEN(node):
    if node is already proven or has no children:
        return
    preferred = 1 if node.turn == root_color else 0
    if any child.proven == preferred:
        node.proven = preferred
    else if node has no untried actions and every child is proven:
        if node.turn == root_color:
            node.proven = maximum child.proven
        else:
            node.proven = minimum child.proven
```

The quantifiers matter:

- At the root player's turn, one proven winning action establishes a win.
- At the opponent's turn, one action that defeats the root player establishes a
  root-player loss at that node.
- To establish a less favorable outcome, all legal choices must be resolved.
  An unexpanded defense can still refute an apparent forced win.
- A draw is not inferred from a mean near 0.5. It must follow from terminal
  rules, forced continuation or sufficient exact child results.

Statistical backup and exact-result backup are separate operations. Neither a
heuristic score nor a sampled rollout outcome is written into `node.proven`.

### Choosing the final root action

```text
BEST_ROOT_ACTION(root):
    eligible = children with visits > 0 or a known proven result
    if any eligible child is proven 1:
        candidates = proven wins
    else if any eligible child is proven 0.5:
        candidates = proven draws
    else if any eligible child is not proven 0:
        candidates = unresolved actions
    else:
        candidates = proven losses

    rank candidates by:
        1. most visits
        2. highest proven value, or mean when unresolved
        3. highest incoming-move material safety
        4. smallest UCI notation
    return the first candidate's move
```

The draw preference fixes an earlier failure: a guaranteed draw could otherwise
be displaced by a heavily visited but uncertain action. Search can continue
through unresolved alternatives to seek a proven win. If no win is proved, a
known draw wins the final comparison even when an unresolved action has an
optimistic mean. This can sacrifice an unproven winning opportunity for safety.

## 5. Rollouts and static evaluation

### Exchange-aware safety

The safety layer estimates the material effect of a move, including the
opponent's legal captures of **any** friendly piece. It can detect a moved
piece hanging, a piece left behind, and unfavorable capture exchanges.

Piece values are pawn 100, knight 320, bishop 330, rook 500 and queen 900.
Legal generation prevents capturing a king. For exchange ordering, a king is
treated as the most valuable attacker, even though its material score is zero.

```text
IMMEDIATE_GAIN(B, move):
    gain = value of captured piece (pawn for en passant)
    if promotion: gain += value(promoted piece) - value(pawn)
    return gain  # castling gains zero

EXCHANGE_REPLY(B, target):
    check deadline
    recaptures = legal captures onto target
    if no recapture exists: return 0
    reply = least-valued legal attacker
            (prefer larger promotion, then UCI notation to break ties)
    gain = IMMEDIATE_GAIN(B, reply)
    play reply
    continuation = EXCHANGE_REPLY(B, target)
    undo reply
    return max(0, gain - continuation)

SEE(B, capture):
    gain = IMMEDIATE_GAIN(B, capture)
    play capture
    continuation = EXCHANGE_REPLY(B, capture.destination)
    undo capture
    return gain - continuation

MOVE_SAFETY(B, move):
    gain = IMMEDIATE_GAIN(B, move)
    play move
    risk = max(0, SEE(B, reply) for every opponent legal capture)
    undo move
    return gain - risk
```

This is a **SEE-style heuristic**, not a full tactical search. Each exchange
follows one least-valued legal attacker on the same square. Recomputing legal
captures after each push handles pins, safe king captures, opened sliding-piece
rays, en passant and capture promotions. It does not search intermediate quiet
moves, checks on other squares, forks, or every alternative attacker sequence.
The optional-stop assumption is also an approximation when a side is in check.

Examples: capturing a pawn and then losing an unrecapturable queen scores
`100 - 900 = -800`. Trading a bishop for a rook scores `500 - 330 = +170`.
An attacked piece is not automatically classified as lost: recaptures and legal
king restrictions matter. A negative safety estimate never becomes an exact
loss, and all legal moves remain available in the main tree. Exact mates and
proven outcomes keep their existing priority.

The cache holds at most 8192 move/position estimates and resets for each search.
Material safety excludes draw history and counters; actual game termination
continues to use history-aware chess rules. Every pushed move is restored on
normal return or a deadline interruption.

### Move ordering

For legal move `m`, the ordering score is:

```text
priority(m) = 4 * MOVE_SAFETY(B, m)
              + 80 if check
              + 40 if castling
              + 3 * (center(destination) - center(origin))
```

Terms apply together when relevant. Checking does not erase a negative
exchange estimate. With `exchange_safety=False`, the old capture priority
`10 * victim - attacker`, promotion bonus and other ordering terms are retained.

### Guided rollout policy

With default probability 0.15, propose uniformly random legal moves. Otherwise
propose with these weights:

```text
weight(m) = 1 + IMMEDIATE_GAIN(B, m) / 100
```

Evaluate a proposed move's safety. Accept it if safety is nonnegative;
otherwise remove it from the current proposal pool and try again. If all legal
moves have negative safety, sample among the moves with the least estimated
loss. This rejection sampling avoids evaluating every quiet move at every ply.
Both sides use the policy, and the random branch also checks safety.

This is a rollout heuristic, not a ban on sacrifices in the main search.
With `exchange_safety=False`, the original uniform/capture-weighted rollout
policy is used. Rollout sampling remains separate from adversarial tree UCT.

```text
EVALUATE_WITH_ROLLOUT(B):
    for ply from 0 through rollout_depth:
        CHECK_DEADLINE()
        if B is terminal:
            return root-player terminal utility
        if ply == rollout_depth:
            return STATIC(B)
        push a move sampled from the rollout policy
```

The main search's `finally` block removes rollout moves as well as tree moves.
With `rollout_depth=0`, the loop evaluates the current tree leaf immediately.
Selection, expansion and backup still occur, so this remains MCTS with leaf
evaluation even though it has no random playout moves. Randomized expansion
ties can still make the seed affect the resulting search.

### Static position evaluation

For each side, sum material and positional bonuses. The code uses:

| Feature | Bonus or rule |
| --- | --- |
| Pawn | Advancement, centralization and an additional quadratic passed-pawn bonus |
| Knight / bishop | Centralization; knights receive a larger center coefficient |
| Bishop pair | 25 points for at least two bishops |
| Rook | Advancement; 12 points when its file contains no friendly pawn |
| Queen | Small centralization bonus |
| King | Centralization in the endgame; center penalty and adjacent pawn shield otherwise |

The endgame threshold is total non-pawn, non-king material of at most 2600.
The passed-pawn test checks opposing pawns ahead on the same or adjacent files;
these bonuses are simple approximations rather than a complete chess evaluator.

```text
D = score(root_color) - score(opponent)
STATIC(B) = 0.5 + 0.5 * tanh(D / 600)
```

This maps the estimate into [0, 1] for backup alongside terminal rewards. It is
not a calibrated winning probability, and a static value near 1 is not a proof.
The static evaluator itself has no hanging-piece or mobility term. Exchange
safety guides move ordering and sampling instead of changing terminal rewards
or replacing the static rollout endpoint.

## 6. Parameters and budgets

### Constructor and runner defaults

Calling `MCTSImproveAgent()` or selecting the registered agent in the game uses
the constructor defaults. The 999-puzzle runner uses those same search
settings, with a simulation budget of 200 unless overridden on the command line.

| Parameter | Constructor default | 999-runner default for `mcts_improve` | Meaning |
| --- | ---: | ---: | --- |
| `simulations` | 400 | 200 | Maximum completed MCTS iterations per chosen move |
| `rollout_depth` | 30 | 30 | Maximum sampled continuation plies after reaching a tree leaf |
| `exploration` | 1.4 | 1.4 | Strength of the UCT exploration bonus |
| `seed` | `None` | 0 | Local randomness; runner uses the selected trial seed |
| `rollout_randomness` | 0.15 | 0.15 | Probability of uniform sampling rather than weighted sampling |
| `progressive_expansion` | `True` | `True` | Gradually admit actions below the root |
| `widening_base` | 2.0 | 2.0 | Multiplier for the width allowance |
| `widening_exponent` | 0.5 | 0.5 | Rate at which width grows with visits |
| `exchange_safety` | `True` | `True` | Legal exchange estimates for ordering, rollouts and final ties |

### What 200 simulations means

200 is an experimental budget, not a requirement of MCTS. Any positive integer
is accepted. It is a cap per `choose_move` call, not per whole puzzle or game.
A mate-in-three puzzle can involve three separate agent searches.

The budget is distributed across tree paths and root actions; each action does
not receive 200 visits. An immediate exact mate can return with zero completed
iterations, and a proven root result can stop search early. A larger cap permits
more work, but does not guarantee a better move in every position.

Rollout depth is not the depth of the entire search tree. One iteration may
traverse several tree plies and then sample additional rollout plies before
static evaluation. With rollout depth 0, it evaluates the tree leaf directly.

### Time limits, history and reproducibility

- `choose_move(board, time_limit=seconds)` applies both the simulation cap and a
  cooperative deadline. Checks cannot preempt a running library call.
- `time_limit=0` returns a deterministic legal fallback, unless terminal or
  single-legal-move handling already applies. `simulations=0` is invalid.
- A fixed seed is restarted for each move search. It does not alter Python's
  global RNG. Without a deadline, fixed input history, parameters and seed
  reproduce the search in the same environment.
- A deadline can change the number of completed iterations between runs.
- Available repetition history is preserved with `board.copy(stack=True)`.
  FEN alone does not reconstruct earlier positions. Claimable draws are treated
  as terminal by this agent's `claim_draw=True` policy.

The 999-puzzle loader retains the supplied PGN history. It reconciles three
move-number differences before replaying that history, so draw checks that
pop and replay moves preserve coherent counters. It does not edit the input
JSON or invent missing moves.

## 7. Running the agent

### Programmatic use

```python
import chess
from agents.mcts_improve import MCTSImproveAgent

agent = MCTSImproveAgent(
    simulations=200,
    rollout_depth=30,
    progressive_expansion=True,
    exchange_safety=True,
    seed=0,
)
board = chess.Board()
move = agent.choose_move(board)  # optional: time_limit=0.1
print(move)
print(agent.last_stats)
print(agent.last_root_stats)
```

For the unchanged constructor defaults, use `MCTSImproveAgent(seed=0)`.
The GUI's registered `mcts_improve` entry also uses constructor defaults.

### One-command puzzle testing

From the Chess-IntroAI folder, with the project Python environment active:

```powershell
# Test the improved agent on all 999 entries.
python test_agent_999.py --agent mcts_improve --simulations 200

# Compare original and improved agents.
python test_agent_999.py --agent MCTS mcts_improve --simulations 200

# Quick functional check; the complete input is still validated first.
python test_agent_999.py --agent mcts_improve --simulations 50 --limit 10
```

If `import chess` fails, activate your project environment and run
`python -m pip install -r requirements.txt` from the repository root.

See [the reusable runner guide](../benchmarks/README_999.md) for custom agents,
seeds, workers, hard timeouts, output paths and constructor-option files.

### Reading search statistics

| Output | Interpretation |
| --- | --- |
| `iterations` | Completed iterations, at most the simulation cap |
| `expanded_nodes` | Explicit children prepared and attached, including an interrupted iteration's attached node |
| `rollout_plies` | Sampled plies belonging to completed iterations; zero with rollout depth 0 |
| `terminal_rollouts`, `heuristic_cutoffs`, `proven_leaves` | How completed evaluations obtained their values |
| `max_tree_depth` | Attempted explicit-tree depth; excludes rollouts and forced-reply lookahead |
| `progressive_selections` | Descents taken while untried actions remained |
| `discarded_iterations` | Incomplete iterations omitted from visits and sampled-value backup |
| `root_visits`, `root_child_visits` | Completed root and child visit totals |
| `elapsed_s`, `stop_reason` | Observed search duration and why search ended |
| `last_root_stats` | Each action's UCI move, visits, root-player mean, exact result, and mover-perspective `safety_cp` |

Typical stop reasons are `simulations`, `proven`, `deadline`, `terminal`,
`forced` and `fallback`. A direct mate check may produce `proven` with zero
iterations. These counters separate actual completed search work from the
configured upper bound.

## 8. Measured evaluation

### Scoring terminology

The benchmark plays the agent on the solver's turns and the supplied reference
moves on the opponent's turns. A full pass requires the reference solver
sequence, or an accepted immediate solver checkmate. Alternative immediate
mating moves are accepted. The 100-puzzle suite also accepts an immediate
terminal draw in an equality/defense position; matching a nonterminal equality
line does not itself mean the game has ended in a draw.

`reference_deviation` means the agent chose a different legal move and the
reference-line test stopped. It is not automatically a losing move or a lost
game; other continuations are not adjudicated. `first_move_match` includes an
accepted immediate alternative mate, and is different from a full-solution pass.
Crashes, illegal moves, input mutation and timeouts have separate statuses.

### Exchange-safety validation

Before this port, the development revision passed 75 unit and integration
tests, including 13 focused safety tests. Unit-test sources are not included
in this branch. At 200 simulations, rollout depth 30 and seed 42, replaying
the two recorded queen-blunder positions with their full game histories produced:

| Position | Safety disabled | Safety enabled |
| --- | --- | --- |
| Before original agent's move 49 | `Qe7+`, queen capturable | `Qe4`, safety 0 |
| Before improved agent's move 46 | `Qa2+`, queen capturable | `Qa5`, safety 0 |

These are move-level regressions, not completed new games or evidence of a
general win-rate improvement. With safety disabled, six seeded comparisons
matched the preserved previous revision's moves and root statistics.

Safety costs computation. One starting-position timing sample at 200
simulations and 30 rollout plies took 2.49 seconds disabled and 5.25 seconds
enabled. This is a single diagnostic sample, not a controlled timing benchmark.

### Historical 100-puzzle result before exchange safety

The earlier static-evaluation runs used seed 0, 400 simulations, 30 rollout
plies and exploration 1.4:

| Configuration | Full solutions out of 100 |
| --- | ---: |
| Progressive expansion off | 66 |
| Progressive expansion on | 71 |

These historical runs predate the exchange-aware safety layer. They are not
measurements of the current safety-enabled revision. The default constructor
still uses progressive expansion with 30 rollout plies.

### Evaluating this revision on all 999 puzzles

The supplied file contains 999 distinct mating positions: 333 each of mate-in-
one, mate-in-two and mate-in-three. Use the runner above to collect current
full-solution scores, first-move scores, error counts and search times.

See the [recorded comparison](../benchmarks/README.md#recorded-results-original-versus-improved-mcts)
for full-run results from before exchange safety. No fresh 999-puzzle score is
claimed for this revision. Results from a different
agent revision must not be presented as its measured performance. Existing
historical experiment files remain unchanged.

Both agents should receive the same simulation cap and seeds for a budgeted
comparison. This matches iteration caps rather than elapsed computation. Use
unseen positions and completed games for broader playing-strength evaluation.

## 9. Limitations

- A promising Monte Carlo average can hide a precise opponent refutation.
- Same-square exchange estimates can miss intermediate moves, forks, attacks
  elsewhere and attacker-order alternatives. They do not guarantee piece safety.
- Safety-based rollout filtering can miss a sacrifice whose compensation is
  positional or requires a longer forced sequence; the main tree retains it.
- A material and positional estimate can miss a fork, trapped piece or quiet
  defensive idea. The static estimate has no explicit king-confinement term
  for converting king-and-queen endgames.
- Exact proofs cover only the terminal positions, immediate mates, forced chains
  and sufficiently solved children actually reached. A timeout or small budget
  can leave a winning action unresolved.
- A proven draw takes priority over every unproven action. This protects a known
  result but can miss a win that needs more search.
- Winning outcomes all have value 1; there is no mate-distance preference. An
  equally winning alternative can therefore depart from a recorded puzzle line.
- Tree statistics do not persist between moves. Transpositions are not shared.
- Matching simulation caps does not match computational effort: move
  preparation, rollout length and early proofs have different costs. Concurrent benchmark
  timings are descriptive, not a controlled equal-time strength comparison.

When testing a change, hold the dataset, seeds and budget fixed, change one
factor at a time, check all puzzle categories, and record elapsed time. Retain
unseen positions and completed, color-swapped games for broader evaluation.
The current frozen sets were used for parameter comparisons and do not serve
as independent evidence that a tuned setting generalizes to all chess play.

## 10. Code map and further reading

| Implementation location | Topic |
| --- | --- |
| `_Node` | Tree fields, visits, value and proof state |
| `choose_move` | Main search, deadlines, board restoration and statistical backup |
| `_select_child` | Opponent-aware UCT |
| `_can_expand` | Progressive width and root exceptions |
| `_prepare_node`, `_update_proven` | Exact checks and proof propagation |
| `_move_priority`, `_rollout_move` | Expansion ordering and rollout sampling |
| `_capture_gain`, `_exchange_reply`, `_see`, `_move_safety` | Legal capture exchanges and material risk |
| `_static_value`, `evaluate` | Positional estimate and terminal utility |
| `_best_child`, `_finish` | Final move policy and statistics |

Each iteration adds at most one explicit tree child. Nodes keep move lists and
statistics instead of complete board copies. Runtime still includes legal-move
generation, history-aware terminal checks, move ordering, sampled plies and
static evaluation; simulation count alone does not measure total computation.

Use the reusable 999-puzzle benchmark commands above to evaluate agents.
The runner validates legal moves, input history and reported outcomes.

Background references:

- [Kocsis and Szepesvari (2006), Bandit Based Monte-Carlo Planning](https://aima.cs.berkeley.edu/~russell/classes/cs294/s11/readings/Kocsis%2BSzepesvari%3A2006.pdf): UCT and bandit-guided selection.
- [Winands, Bjornsson and Saito (2008), Monte-Carlo Tree Search Solver](https://dke.maastrichtuniversity.nl/m.winands/documents/uctloa.pdf): propagating solved outcomes in an MCTS search.
