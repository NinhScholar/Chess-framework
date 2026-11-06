"""
Alpha-Beta Pruning Agent (Search & Pruning Optimization Approach).

Key techniques implemented:
1. Negamax formulation with Alpha-Beta Pruning.
2. Quiescence Search at leaf nodes to eliminate the Horizon Effect.
3. Move Ordering:
   - PV / Hash move from previous iteration.
   - MVV-LVA (Most Valuable Victim - Least Valuable Attacker) for captures & promotions.
   - Killer Move Heuristic for quiet moves that produced beta cutoffs.
   - Checks priority.
   - History Heuristic for non-capture moves.
4. Iterative Deepening Search (IDS) with adaptive time control.
5. Material and Piece-Square Table (PST) evaluation.
"""

import time
import chess
from .agent import Agent

# ==============================================================================
# HẰNG SỐ & ĐỊNH GIÁ QUÂN CỜ (CENTIPAWNS)
# ==============================================================================
MATE_SCORE = 100_000
INFINITY = 1_000_000

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

# ==============================================================================
# PIECE-SQUARE TABLES (Từ góc nhìn quân Trắng, rank 1 -> rank 8)
# A1 = 0, H1 = 7 (Hàng đầu phe Trắng); A8 = 56, H8 = 63 (Hàng đầu phe Đen)
# ==============================================================================
PAWN_PST = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10,-20,-20, 10, 10,  5,
     5, -5,-10,  0,  0,-10, -5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
     0,  0,  0,  0,  0,  0,  0,  0,
]

KNIGHT_PST = [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50,
]

BISHOP_PST = [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20,
]

ROOK_PST = [
     0,  0,  0,  5,  5,  0,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

QUEEN_PST = [
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -10,  5,  5,  5,  5,  5,  0,-10,
     0,  0,  5,  5,  5,  5,  0, -5,
    -5,  0,  5,  5,  5,  5,  0, -5,
   -10,  0,  5,  5,  5,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20,
]

KING_PST = [
    20, 30, 10,  0,  0, 10, 30, 20,
    20, 20,  0,  0,  0,  0, 20, 20,
   -10,-20,-20,-20,-20,-20,-20,-10,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
]

PST_MAP = {
    chess.PAWN: PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.QUEEN: QUEEN_PST,
    chess.KING: KING_PST,
}


def evaluate_board(board: chess.Board) -> int:
    """
    Hàm lượng giá tĩnh nhanh: Vật chất (Material) + Vị trí (Piece-Square Table).
    Trả về điểm số theo góc nhìn của bên đang tới lượt đi (board.turn).
    """
    white_score = 0
    black_score = 0

    for square, piece in board.piece_map().items():
        val = PIECE_VALUES[piece.piece_type]
        pst = PST_MAP[piece.piece_type]

        if piece.color == chess.WHITE:
            white_score += val + pst[square]
        else:
            # Đối với quân Đen, lật ngược chỉ số square theo trục đứng (square ^ 56)
            black_score += val + pst[square ^ 56]

    diff = white_score - black_score
    return diff if board.turn == chess.WHITE else -diff


# ==============================================================================
# AGENT CLASS: ALPHA-BETA SEARCH (APPROACH 1)
# ==============================================================================
class AlphaBetaSearchAgent(Agent):
    """
    Agent cờ vua sử dụng Alpha-Beta Pruning tối ưu hóa theo chiều sâu tìm kiếm.
    Bao gồm:
      - Negamax Alpha-Beta Pruning
      - Quiescence Search (khử Horizon Effect)
      - Move Ordering (MVV-LVA, Killer Moves, History Heuristic, Checks)
      - Iterative Deepening Search với Time Control
    """
    name = "AlphaBeta_Search"

    def __init__(self, max_depth: int = 5, default_time: float = 1.0):
        self.max_depth = max_depth
        self.default_time = default_time

        # Thống kê đo đạc hiệu năng
        self.nodes_visited = 0
        self.quiescence_nodes = 0
        self.beta_cutoffs = 0
        self.last_info = {}

        # Heuristics tables
        self.killer_moves = {}    # killer_moves[ply] = [move1, move2]
        self.history_table = {}   # history_table[(color, from_sq, to_sq)] = score

    def on_game_start(self, color):
        self.killer_moves.clear()
        self.history_table.clear()

    # --------------------------------------------------------------------------
    # SẮP XẾP NƯỚC ĐI (MOVE ORDERING)
    # --------------------------------------------------------------------------
    def _score_move(self, board: chess.Board, move: chess.Move, ply: int, pv_move: chess.Move | None) -> int:
        """Chấm điểm ưu tiên cho nước đi để duyệt các nước hứa hẹn nhất trước."""
        # 1. Nước đi tốt nhất từ tầng trước (PV move)
        if pv_move and move == pv_move:
            return 10_000_000

        # 2. Bắt quân (Captures) theo nguyên lý MVV-LVA
        if board.is_capture(move):
            attacker = board.piece_at(move.from_square)
            attacker_type = attacker.piece_type if attacker else chess.PAWN
            attacker_val = PIECE_VALUES[attacker_type]

            # Xác định quân bị bắt (bao gồm cả en-passant)
            victim = board.piece_at(move.to_square)
            if victim:
                victim_val = PIECE_VALUES[victim.piece_type]
            elif board.is_en_passant(move):
                victim_val = PIECE_VALUES[chess.PAWN]
            else:
                victim_val = 100

            # Phong cấp
            promo_bonus = PIECE_VALUES[move.promotion] if move.promotion else 0
            # Điểm: Ưu tiên nạn nhân giá trị cao bị ăn bởi quân giá trị thấp
            return 1_000_000 + (victim_val * 10 - attacker_val) + promo_bonus

        # 3. Phong cấp không bắt quân
        if move.promotion:
            return 900_000 + PIECE_VALUES[move.promotion]

        # 4. Killer Moves (nước không bắt quân từng gây cắt tỉa ở cùng độ sâu)
        killers = self.killer_moves.get(ply, [])
        if len(killers) > 0 and move == killers[0]:
            return 800_000
        if len(killers) > 1 and move == killers[1]:
            return 700_000

        # 5. Nước chiếu vua (Checks)
        if board.gives_check(move):
            return 600_000

        # 6. History Heuristic (nước đi từng gây cắt tỉa nhiều lần trong quá khứ)
        history_key = (board.turn, move.from_square, move.to_square)
        return self.history_table.get(history_key, 0)

    def _order_moves(self, board: chess.Board, moves, ply: int, pv_move: chess.Move | None = None):
        """Sắp xếp danh sách nước đi theo thứ tự điểm ưu tiên giảm dần."""
        return sorted(moves, key=lambda m: self._score_move(board, m, ply, pv_move), reverse=True)

    def _order_captures(self, board: chess.Board, captures):
        """Sắp xếp riêng các nước ăn quân cho Quiescence Search theo MVV-LVA."""
        def capture_score(move):
            attacker = board.piece_at(move.from_square)
            attacker_val = PIECE_VALUES[attacker.piece_type] if attacker else 100
            victim = board.piece_at(move.to_square)
            victim_val = PIECE_VALUES[victim.piece_type] if victim else 100
            return victim_val * 10 - attacker_val

        return sorted(captures, key=capture_score, reverse=True)

    # --------------------------------------------------------------------------
    # TÌM KIẾM TĨNH LẶNG (QUIESCENCE SEARCH)
    # --------------------------------------------------------------------------
    def _quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int, deadline: float | None) -> int:
        """
        Duyệt tiếp các nước ăn quân để triệt tiêu Horizon Effect.
        Không đánh giá tĩnh khi bàn cờ còn đang trong chuỗi bắt quân.
        """
        self.nodes_visited += 1
        self.quiescence_nodes += 1

        if deadline and time.perf_counter() >= deadline:
            raise TimeoutError()

        # Kiểm tra điều kiện hòa / hết ván
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0

        # Stand-pat: điểm lượng giá tĩnh tại trạng thái hiện thời
        stand_pat = evaluate_board(board)

        # Cắt tỉa nếu đối thủ đã có nhánh tốt hơn
        if stand_pat >= beta:
            return beta

        if stand_pat > alpha:
            alpha = stand_pat

        # Delta Pruning: nếu ăn cả quân lớn nhất (Hậu) mà vẫn không cải thiện được alpha
        # thì bỏ qua mở rộng (trừ khi đang tàn cuộc)
        BIGGEST_PIECE_VALUE = 900
        if stand_pat < alpha - BIGGEST_PIECE_VALUE:
            return alpha

        # Chỉ duyệt các nước ăn quân (hoặc nước chiếu / phong cấp)
        captures = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
        if not captures:
            return alpha

        ordered_captures = self._order_captures(board, captures)

        for move in ordered_captures:
            board.push(move)
            score = -self._quiescence(board, -beta, -alpha, ply + 1, deadline)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    # --------------------------------------------------------------------------
    # THUẬT TOÁN NEGAMAX ALPHA-BETA
    # --------------------------------------------------------------------------
    def _negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int,
                 deadline: float | None, pv_move: chess.Move | None = None) -> tuple[int, chess.Move | None]:
        self.nodes_visited += 1

        if deadline and time.perf_counter() >= deadline:
            raise TimeoutError()

        # Kiểm tra kết thúc ván
        if board.is_checkmate():
            return -MATE_SCORE + ply, None
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0, None

        # Chạm đáy độ sâu -> chuyển sang tìm kiếm tĩnh lặng
        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply, deadline), None

        ordered_moves = self._order_moves(board, board.legal_moves, ply, pv_move)
        if not ordered_moves:
            if board.is_check():
                return -MATE_SCORE + ply, None
            return 0, None

        best_score = -INFINITY
        best_move = ordered_moves[0]

        for move in ordered_moves:
            board.push(move)
            try:
                score, _ = self._negamax(board, depth - 1, -beta, -alpha, ply + 1, deadline)
                score = -score
            finally:
                board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

            if alpha >= beta:
                self.beta_cutoffs += 1
                # Cập nhật Killer Moves và History Heuristic cho nước không bắt quân
                if not board.is_capture(move):
                    if ply not in self.killer_moves:
                        self.killer_moves[ply] = [move]
                    elif move != self.killer_moves[ply][0]:
                        self.killer_moves[ply] = [move, self.killer_moves[ply][0]]

                    h_key = (board.turn, move.from_square, move.to_square)
                    self.history_table[h_key] = self.history_table.get(h_key, 0) + depth * depth
                break

        return best_score, best_move

    # --------------------------------------------------------------------------
    # HÀM CHỌN NƯỚC ĐI CHÍNH (ITERATIVE DEEPENING SEARCH)
    # --------------------------------------------------------------------------
    def choose_move(self, board: chess.Board, time_limit: float | None = None) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            return legal_moves[0]

        start_time = time.perf_counter()
        effective_time = time_limit if time_limit is not None else self.default_time
        deadline = start_time + effective_time * 0.95 if effective_time else None

        self.nodes_visited = 0
        self.quiescence_nodes = 0
        self.beta_cutoffs = 0

        best_move_overall = legal_moves[0]
        completed_depth = 0
        pv_move = None

        # Iterative Deepening: duyệt từ depth 1 lên max_depth
        for depth in range(1, self.max_depth + 1):
            try:
                score, move = self._negamax(
                    board=board,
                    depth=depth,
                    alpha=-INFINITY,
                    beta=INFINITY,
                    ply=0,
                    deadline=deadline,
                    pv_move=pv_move
                )
                if move:
                    best_move_overall = move
                    pv_move = move
                    completed_depth = depth

                # Nếu tìm ra nước chiếu hết cưỡng bức thì không cần tìm sâu hơn
                if score >= MATE_SCORE - 100 or score <= -MATE_SCORE + 100:
                    break

            except TimeoutError:
                # Hết thời gian: ngắt tìm kiếm ở độ sâu dở dang, dùng kết quả độ sâu trước
                break

        elapsed = time.perf_counter() - start_time
        nps = int(self.nodes_visited / elapsed) if elapsed > 0 else 0
        b_eff = (self.nodes_visited ** (1.0 / max(1, completed_depth))) if self.nodes_visited > 0 else 0.0

        self.last_info = {
            "depth": completed_depth,
            "nodes": self.nodes_visited,
            "q_nodes": self.quiescence_nodes,
            "cutoffs": self.beta_cutoffs,
            "time": elapsed,
            "nps": nps,
            "b_eff": round(b_eff, 2),
        }

        return best_move_overall
