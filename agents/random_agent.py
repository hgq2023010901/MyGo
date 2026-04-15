"""随机落子围棋智能体。

从合法棋步中随机选择下一手，可用于基线对战和规则验证。
"""

import random
from dlgo import GameState, Move

__all__ = ["RandomAgent"]


class RandomAgent:
    """
    随机落子智能体。

    从所有合法棋步中均匀随机选择，包括：
    - 正常落子
    - 停一手 (pass)
    - 认输 (resign)
    """

    def __init__(self):
        """初始化随机智能体（无需特殊参数）"""
        self.rng = random.Random()

    def select_move(self, game_state: GameState) -> Move:
        """
        选择随机合法棋步

        Args:
            game_state: 当前游戏状态

        Returns:
            随机选择的合法 Move
        """
        legal_moves = game_state.legal_moves()
        non_resign_moves = [move for move in legal_moves if not move.is_resign]
        if non_resign_moves:
            return self.rng.choice(non_resign_moves)
        return self.rng.choice(legal_moves)


# 便捷函数（向后兼容 play.py）
def random_agent(game_state: GameState) -> Move:
    """函数接口，兼容 play.py 的调用方式"""
    agent = RandomAgent()
    return agent.select_move(game_state)