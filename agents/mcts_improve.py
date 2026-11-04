"""Progressively expanded UCT with guided rollouts and exact proofs.

Values are always for the player at the search root: win=1, draw=0.5,
loss=0. Opponent nodes minimize that value. Proven results come only from
terminal positions, an explicitly checked mate in one, forced replies, or solved children;
rollout estimates never constitute a proof.

Each move starts a fresh lightweight tree. One private board, including its
repetition history, follows selection and simulation using push/pop. A fixed
seed and simulation cap reproduce a search; wall-clock cutoffs may not.
"""

import math
import random
import time
from dataclasses import dataclass, field

import chess

from .agent import Agent


_PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


class _SearchTimeout(Exception):
    pass


@dataclass(eq=False, slots=True)
class _Node:
    turn: chess.Color
    move: chess.Move | None = None
    parent: "_Node | None" = None
    children: list = field(default_factory=list)
    untried: list = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    proven: float | None = None
    mate_move: chess.Move | None = None
    safety: float = 0.0

    @property
    def mean(self):
        return self.value / self.visits if self.visits else 0.5


class MCTSImproveAgent(Agent):
    """UCT plus prioritized expansion, capture-guided rollouts and solved nodes.

    Both the simulation cap and an optional time_limit (seconds) apply. A
    zero time budget returns a deterministic legal fallback; terminal inputs return
    None. seed is local to each search and never alters Python's global RNG.

    last_stats counts completed iterations and their rollout plies only.
    expanded_nodes excludes the root but includes prepared nodes from an
    interrupted rollout. Exact mate checks can finish before any iteration.
    last_root_stats exposes visits, root-perspective means and exact proofs.
    max_tree_depth and progressive_selections describe attempted traversal.
    Nonterminal rollout cutoffs use the static material/placement estimate.
    Disable progressive_expansion to expand every untried action before UCT.
    exchange_safety guides ordering, rollout sampling and final ties using
    legal capture exchanges; it never changes terminal utilities or proofs.
    """

    name = "mcts_improve"

    def __init__(self, simulations=400, rollout_depth=30, exploration=1.4,
                 seed=None, rollout_randomness=0.15, progressive_expansion=True,
                 widening_base=2.0, widening_exponent=0.5,
                 exchange_safety=True):
        if type(simulations) is not int or simulations <= 0:
            raise ValueError("simulations must be a positive integer")
        if type(rollout_depth) is not int or rollout_depth < 0:
            raise ValueError("rollout_depth must be a nonnegative integer")
        if not math.isfinite(exploration) or exploration < 0:
            raise ValueError("exploration must be finite and nonnegative")
        if seed is not None and type(seed) is not int:
            raise ValueError("seed must be an integer or None")
        if not math.isfinite(rollout_randomness) or not 0 <= rollout_randomness <= 1:
            raise ValueError("rollout_randomness must lie in [0, 1]")
        if type(progressive_expansion) is not bool:
            raise ValueError("progressive_expansion must be a boolean")
        if not math.isfinite(widening_base) or widening_base <= 0:
            raise ValueError("widening_base must be finite and positive")
        if not math.isfinite(widening_exponent) or not 0 < widening_exponent < 1:
            raise ValueError("widening_exponent must lie strictly between 0 and 1")
        if type(exchange_safety) is not bool:
            raise ValueError("exchange_safety must be a boolean")
        self.simulations = simulations
        self.rollout_depth = rollout_depth
        self.exploration = exploration
        self.seed = seed
        self.rollout_randomness = rollout_randomness
        self.progressive_expansion = progressive_expansion
        self.widening_base = widening_base
        self.widening_exponent = widening_exponent
        self.exchange_safety = exchange_safety
        self._safety_cache = {}
        self.last_stats = {}
        self.last_root_stats = []

    @staticmethod
    def _check_deadline(deadline):
        if deadline is not None and time.perf_counter() >= deadline:
            raise _SearchTimeout

    @staticmethod
    def _utility(outcome, color):
        if outcome.winner is None:
            return 0.5
        return float(outcome.winner == color)

    @staticmethod
    def _center(square):
        return (3.5 - abs(3.5 - chess.square_file(square))
                + 3.5 - abs(3.5 - chess.square_rank(square)))

    def _static_value(self, board, color):
        """Bounded material/placement heuristic, not a win probability."""
        scores = {chess.WHITE: 0.0, chess.BLACK: 0.0}
        heavy_material = sum(
            _PIECE_VALUES[piece_type] * len(board.pieces(piece_type, side))
            for side in chess.COLORS
            for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
        )
        for side in chess.COLORS:
            for piece_type, points in _PIECE_VALUES.items():
                for square in board.pieces(piece_type, side):
                    rank = chess.square_rank(square)
                    advance = rank if side else 7 - rank
                    center = self._center(square)
                    bonus = 0.0
                    if piece_type == chess.PAWN:
                        bonus = 7 * max(0, advance - 1) + 2 * center
                        file = chess.square_file(square)
                        opposing_pawns = board.pieces(chess.PAWN, not side)
                        blocked = any(
                            abs(chess.square_file(other) - file) <= 1
                            and (chess.square_rank(other) > rank if side
                                 else chess.square_rank(other) < rank)
                            for other in opposing_pawns
                        )
                        if not blocked:
                            bonus += 3 * max(0, advance - 1) ** 2
                    elif piece_type in (chess.KNIGHT, chess.BISHOP):
                        bonus = center * (8 if piece_type == chess.KNIGHT else 5)
                    elif piece_type == chess.ROOK:
                        bonus = 2 * advance
                        if not (board.pawns & board.occupied_co[side]
                                & chess.BB_FILES[chess.square_file(square)]):
                            bonus += 12
                    elif piece_type == chess.QUEEN:
                        bonus = 2 * center
                    elif piece_type == chess.KING:
                        if heavy_material <= 2600:
                            bonus = 10 * center
                        else:
                            bonus = -8 * center
                            shield_rank = rank + (1 if side else -1)
                            if 0 <= shield_rank < 8:
                                for file in range(max(0, chess.square_file(square) - 1),
                                                  min(8, chess.square_file(square) + 2)):
                                    if board.piece_at(chess.square(file, shield_rank)) == chess.Piece(chess.PAWN, side):
                                        bonus += 12
                    scores[side] += points + bonus
            if len(board.pieces(chess.BISHOP, side)) >= 2:
                scores[side] += 25
        difference = scores[color] - scores[not color]
        return 0.5 + 0.5 * math.tanh(difference / 600.0)

    def evaluate(self, board, color):
        outcome = board.outcome(claim_draw=True)
        return (self._utility(outcome, color) if outcome is not None
                else self._static_value(board, color))

    @staticmethod
    def _capture_gain(board, move):
        """Immediate material gain, including en passant and promotion."""
        if board.is_castling(move):
            return 0
        victim = (chess.PAWN if board.is_en_passant(move)
                  else board.piece_type_at(move.to_square))
        gain = _PIECE_VALUES.get(victim, 0)
        if move.promotion:
            gain += _PIECE_VALUES[move.promotion] - _PIECE_VALUES[chess.PAWN]
        return gain

    def _exchange_reply(self, board, target, deadline):
        """SEE-style recaptures: legal least-valued attackers, optional stop.

        Legal generation accounts for pins, king safety and newly opened rays.
        Choosing one least-valued attacker is a heuristic, not a tactical proof;
        intermediate checks and captures on other squares are not searched.
        """
        self._check_deadline(deadline)
        if board.piece_type_at(target) in (None, chess.KING):
            return 0
        captures = list(board.generate_legal_captures(to_mask=chess.BB_SQUARES[target]))
        if not captures:
            return 0
        move = min(captures, key=lambda m: (
            100000 if board.piece_type_at(m.from_square) == chess.KING
            else _PIECE_VALUES[board.piece_type_at(m.from_square)],
            -_PIECE_VALUES.get(m.promotion, 0), m.uci()))
        gain = self._capture_gain(board, move)
        board.push(move)
        try:
            return max(0, gain - self._exchange_reply(board, target, deadline))
        finally:
            board.pop()

    def _see(self, board, move, deadline=None):
        """Net material from a legal initial capture and same-square exchanges."""
        self._check_deadline(deadline)
        gain = self._capture_gain(board, move)
        board.push(move)
        try:
            return gain - self._exchange_reply(board, move.to_square, deadline)
        finally:
            board.pop()

    def _move_safety(self, board, move, deadline=None):
        """Mover's gain minus the opponent's best profitable legal exchange.

        All opponent captures are considered, including pieces left behind.
        Scores guide sampling only; they never establish node.proven.
        """
        self._check_deadline(deadline)
        key = (board.pawns, board.knights, board.bishops, board.rooks,
               board.queens, board.kings, board.occupied_co[chess.WHITE],
               board.occupied_co[chess.BLACK], board.turn,
               board.castling_rights, board.ep_square, board.promoted, move)
        if key in self._safety_cache:
            return self._safety_cache[key]
        gain = self._capture_gain(board, move)
        board.push(move)
        try:
            risk = 0
            captures = sorted(board.generate_legal_captures(),
                              key=lambda reply: self._capture_gain(board, reply),
                              reverse=True)
            for reply in captures:
                self._check_deadline(deadline)
                # SEE cannot exceed the immediate gain: recapturing can be
                # declined. Lower-gain replies cannot improve the current risk.
                if self._capture_gain(board, reply) <= risk:
                    break
                risk = max(risk, self._see(board, reply, deadline))
            score = gain - risk
        finally:
            board.pop()
        # Bounded, per-search cache; material safety does not depend on clocks.
        if len(self._safety_cache) >= 8192:
            self._safety_cache.clear()
        self._safety_cache[key] = score
        return score

    def _move_priority(self, board, move, gives_check, deadline=None):
        victim = board.piece_type_at(move.to_square)
        if board.is_en_passant(move):
            victim = chess.PAWN
        attacker = board.piece_type_at(move.from_square)
        priority = 0.0
        if victim is not None:
            priority += 10 * _PIECE_VALUES[victim] - _PIECE_VALUES[attacker]
        if move.promotion:
            priority += _PIECE_VALUES[move.promotion]
        if self.exchange_safety:
            priority = 4 * self._move_safety(board, move, deadline)
        if gives_check:
            priority += 80
        if board.is_castling(move):
            priority += 40
        priority += 3 * (self._center(move.to_square) - self._center(move.from_square))
        return priority

    def _prepare_node(self, board, node, color, rng, deadline, forced_depth=2):
        self._check_deadline(deadline)
        outcome = board.outcome(claim_draw=True)
        self._check_deadline(deadline)
        if outcome is not None:
            node.proven = self._utility(outcome, color)
            return
        moves = list(board.legal_moves)
        # A single legal reply introduces no adversarial choice. Look through
        # at most two such replies for an exact terminal/mate-in-one result.
        # This is not a general minimax search, and no heuristic is a proof.
        if len(moves) == 1 and forced_depth:
            board.push(moves[0])
            try:
                forced_child = _Node(turn=board.turn)
                self._prepare_node(board, forced_child, color, rng, deadline,
                                   forced_depth - 1)
            finally:
                board.pop()
            if forced_child.proven is not None:
                node.proven = forced_child.proven
                return
        ranked = []
        for move in moves:
            self._check_deadline(deadline)
            gives_check = board.gives_check(move)
            if gives_check:
                board.push(move)
                try:
                    is_mate = board.is_checkmate()
                finally:
                    board.pop()
                if is_mate:
                    node.proven = float(node.turn == color)
                    node.mate_move = move
                    return
            ranked.append((self._move_priority(board, move, gives_check, deadline),
                           rng.random(), move.uci(), move))
        # Pop the highest-priority move; random keys break equal-priority ties.
        node.untried = [item[3] for item in sorted(ranked)]

    @staticmethod
    def _update_proven(node, color):
        if node.proven is not None or not node.children:
            return
        preferred = float(node.turn == color)
        if any(child.proven == preferred for child in node.children):
            node.proven = preferred
        elif not node.untried and all(child.proven is not None for child in node.children):
            values = [child.proven for child in node.children]
            node.proven = max(values) if node.turn == color else min(values)

    def _select_child(self, node, color):
        # Exact outcomes need no more sampling. Final selection must compare
        # those proofs with estimates rather than treating low visits as weak.
        unresolved = [child for child in node.children if child.proven is None]

        def uct(child):
            if not child.visits:
                return math.inf
            exploitation = child.mean if node.turn == color else 1 - child.mean
            return exploitation + self.exploration * math.sqrt(
                math.log(max(1, node.visits)) / child.visits)

        return max(unresolved, key=uct)

    def _can_expand(self, node):
        if not node.untried:
            return False
        # Cover every root action before narrowing effort deeper in the tree.
        if not self.progressive_expansion or node.parent is None:
            return True
        # Solved children need no further sampling. Admit another legal action
        # if none remain selectable, even if the width allowance was exhausted.
        if not any(child.proven is None for child in node.children):
            return True
        allowance = max(1, int(self.widening_base
                               * max(1, node.visits) ** self.widening_exponent))
        return len(node.children) < allowance

    def _rollout_move(self, board, moves, rng, deadline=None):
        if self.exchange_safety:
            # Rejection sampling avoids checking every quiet action at every
            # rollout ply. Conditional on passing safety, sampling remains
            # uniform or weighted by immediate capture/promotion gain.
            remaining = list(moves)
            weights = [1 + self._capture_gain(board, move) / 100 for move in remaining]
            uniform = rng.random() < self.rollout_randomness
            rejected = []
            while remaining:
                self._check_deadline(deadline)
                index = (rng.randrange(len(remaining)) if uniform
                         else rng.choices(range(len(remaining)), weights=weights, k=1)[0])
                move = remaining.pop(index)
                weights.pop(index)
                safety = self._move_safety(board, move, deadline)
                if safety >= 0:
                    return move
                rejected.append((move, safety))
            # A forced material loss must not leave the policy without a move.
            best = max(score for _, score in rejected)
            return rng.choice([move for move, score in rejected if score == best])
        if rng.random() < self.rollout_randomness:
            return rng.choice(moves)
        weights = []
        for move in moves:
            weight = 1.0
            if board.is_capture(move):
                victim = (chess.PAWN if board.is_en_passant(move)
                          else board.piece_type_at(move.to_square))
                attacker = board.piece_type_at(move.from_square)
                weight += max(1.0, (10 * _PIECE_VALUES[victim]
                                   - _PIECE_VALUES[attacker]) / 100)
            if move.promotion:
                weight += _PIECE_VALUES[move.promotion] / 50
            weights.append(weight)
        return rng.choices(moves, weights=weights, k=1)[0]

    def _rollout(self, board, color, rng, deadline):
        for ply in range(self.rollout_depth + 1):
            self._check_deadline(deadline)
            outcome = board.outcome(claim_draw=True)
            self._check_deadline(deadline)
            if outcome is not None:
                return self._utility(outcome, color), ply, "terminal_rollouts"
            if ply == self.rollout_depth:
                value = self._static_value(board, color)
                self._check_deadline(deadline)
                return value, ply, "heuristic_cutoffs"
            moves = list(board.legal_moves)
            board.push(self._rollout_move(board, moves, rng, deadline))

    @staticmethod
    def _best_child(root):
        # An exact proof remains valid even if a deadline interrupted the
        # iteration before its visits were backed up.
        candidates = [child for child in root.children
                      if child.visits or child.proven is not None]
        winning = [child for child in candidates if child.proven == 1.0]
        if winning:
            candidates = winning
        else:
            drawing = [child for child in candidates if child.proven == 0.5]
            if drawing:
                # Solved draws are explored once; unresolved actions may have
                # hundreds of visits. Prefer the guaranteed half-point unless
                # search actually proves a win, even if estimates are optimistic.
                candidates = drawing
            else:
                nonlosing = [child for child in candidates if child.proven != 0.0]
                if nonlosing:
                    candidates = nonlosing

        def rank(child):
            value = child.proven if child.proven is not None else child.mean
            return -child.visits, -value, -child.safety, child.move.uci()

        return min(candidates, key=rank) if candidates else None

    def _finish(self, started, move, reason, root=None):
        self.last_stats["elapsed_s"] = time.perf_counter() - started
        self.last_stats["stop_reason"] = reason
        if root is not None:
            self.last_stats["root_visits"] = root.visits
            self.last_stats["root_child_visits"] = sum(child.visits for child in root.children)
            self.last_root_stats = [
                {"move": child.move.uci(), "visits": child.visits,
                 "mean": child.mean, "proven": child.proven,
                 "safety_cp": child.safety}
                for child in root.children
            ]
        return move

    def choose_move(self, board: chess.Board, time_limit=None) -> chess.Move | None:
        started = time.perf_counter()
        if time_limit is not None and (not math.isfinite(time_limit) or time_limit < 0):
            raise ValueError("time_limit must be finite and nonnegative, or None")
        deadline = None if time_limit is None else started + time_limit
        self.last_stats = {
            "iterations": 0, "expanded_nodes": 0, "rollout_plies": 0,
            "terminal_rollouts": 0, "heuristic_cutoffs": 0,
            "proven_leaves": 0, "discarded_iterations": 0,
            "root_visits": 0, "root_child_visits": 0,
            "max_tree_depth": 0, "progressive_selections": 0,
        }
        self.last_root_stats = []
        # Terminal rules take precedence, including draws with legal moves.
        if board.outcome(claim_draw=True) is not None:
            return self._finish(started, None, "terminal")
        moves = list(board.legal_moves)
        fallback = min(moves, key=lambda move: move.uci())
        if len(moves) == 1:
            return self._finish(started, fallback, "forced")
        private = board.copy(stack=True)
        stack_size = len(private.move_stack)
        color = private.turn
        rng = random.Random(self.seed)
        self._safety_cache.clear()
        root = _Node(turn=color)
        try:
            self._prepare_node(private, root, color, rng, deadline)
        except _SearchTimeout:
            return self._finish(started, fallback, "fallback", root)
        reason = "simulations"
        while self.last_stats["iterations"] < self.simulations and root.proven is None:
            node = root
            path = [root]
            try:
                self._check_deadline(deadline)
                while node.proven is None:
                    self._check_deadline(deadline)
                    if self._can_expand(node):
                        move = node.untried[-1]
                        safety = (self._move_safety(private, move, deadline)
                                  if self.exchange_safety else 0.0)
                        private.push(move)
                        child = _Node(turn=private.turn, move=move, parent=node,
                                      safety=safety)
                        self._prepare_node(private, child, color, rng, deadline)
                        node.untried.pop()
                        node.children.append(child)
                        self.last_stats["expanded_nodes"] += 1
                        node = child
                        path.append(child)
                        self.last_stats["max_tree_depth"] = max(
                            self.last_stats["max_tree_depth"], len(path) - 1)
                        break
                    if node.untried:
                        self.last_stats["progressive_selections"] += 1
                    node = self._select_child(node, color)
                    private.push(node.move)
                    path.append(node)
                    self.last_stats["max_tree_depth"] = max(
                        self.last_stats["max_tree_depth"], len(path) - 1)
                if node.proven is not None:
                    result, plies, kind = node.proven, 0, "proven_leaves"
                else:
                    result, plies, kind = self._rollout(private, color, rng, deadline)
                self._check_deadline(deadline)
            except _SearchTimeout:
                self.last_stats["discarded_iterations"] += 1
                reason = "deadline"
                break
            finally:
                while len(private.move_stack) > stack_size:
                    private.pop()
            for visited in reversed(path):
                visited.visits += 1
                visited.value += result
                self._update_proven(visited, color)
            self.last_stats["iterations"] += 1
            self.last_stats["rollout_plies"] += plies
            self.last_stats[kind] += 1
        if root.proven is not None:
            reason = "proven"
        if root.mate_move is not None:
            return self._finish(started, root.mate_move, reason, root)
        best = self._best_child(root)
        if best is None:
            return self._finish(started, fallback, "fallback", root)
        return self._finish(started, best.move, reason, root)
