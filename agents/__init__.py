from .human_agent import HumanAgent
from .random_agent import RandomAgent
from .mcts_agent import MCTSAgent
from .mcts_improve import MCTSImproveAgent
from .stockfish_agent import StockfishAgent
from .promotion_test_agent import PromotionTestAgent
from .stockfish_agent_report import StockfishAgentReport

AGENT_FACTORIES = {
    "Người chơi": HumanAgent,
    "Random": RandomAgent,
    "MCTS": lambda: MCTSAgent(),
    "mcts_improve": MCTSImproveAgent,
    "Test phong tốt": PromotionTestAgent,
    "Stockfish": lambda: StockfishAgent(level=20),
    "StockfishReport": lambda: StockfishAgentReport(level=20)
} 