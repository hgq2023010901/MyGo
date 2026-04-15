"""
MCTS (蒙特卡洛树搜索) 智能体实现。

实现包含：选择、扩展、模拟、反向传播。
同时加入启发式走子排序与有限深度 rollout 两种加速策略。
"""

import math
import random

from dlgo.gotypes import Player, Point
from dlgo.goboard import GameState, Move
from dlgo.scoring import compute_game_result

__all__ = ["MCTSAgent"]



class MCTSNode:
    """
    MCTS 树节点。


    属性：
        game_state: 当前局面
        parent: 父节点（None 表示根节点）
        children: 子节点列表
        visit_count: 访问次数
        value_sum: 累积价值（胜场数）
        prior: 先验概率（来自策略网络，可选）
    """

    def __init__(self, game_state, parent=None, prior=1.0):
        self.game_state = game_state
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.untried_moves = self._ordered_moves()
        self.move = None

    @property
    def value(self):
        """计算平均价值 = value_sum / visit_count，防止除零。"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_leaf(self):
        """是否为叶节点（未展开）。"""
        return len(self.children) == 0

    def is_terminal(self):
        """是否为终局节点。"""
        return self.game_state.is_over()

    def best_child(self, c=1.414):
        """
        选择最佳子节点（UCT 算法）。

        UCT = value + c * sqrt(ln(parent_visits) / visits)

        Args:
            c: 探索常数（默认 sqrt(2)）

        Returns:
            最佳子节点
        """
        best_score = None
        best_children = []
        parent_visits = max(1, self.visit_count)

        for child in self.children:
            if child.visit_count == 0:
                score = float("inf")
            else:
                exploitation = 1.0 - child.value
                exploration = c * math.sqrt(math.log(parent_visits) / child.visit_count)
                score = exploitation + exploration

            if best_score is None or score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)

        return random.choice(best_children)

    def expand(self):
        """
        展开节点：为所有合法棋步创建子节点。

        Returns:
            新创建的子节点（用于后续模拟）
        """
        if not self.untried_moves:
            return None

        move = self.untried_moves.pop(0)
        next_state = self.game_state.apply_move(move)
        child = MCTSNode(next_state, parent=self)
        child.move = move
        self.children.append(child)
        return child

    def backup(self, value):
        """
        反向传播：更新从当前节点到根节点的统计。

        Args:
            value: 从当前局面模拟得到的结果（1=胜，0=负，0.5=和）
        """
        node = self
        current_value = value
        while node is not None:
            node.visit_count += 1
            node.value_sum += current_value
            current_value = 1.0 - current_value
            node = node.parent

    def _ordered_moves(self):
        moves = self.game_state.legal_moves()
        legal_moves = [move for move in moves if not move.is_resign]
        if legal_moves:
            return sorted(
                legal_moves,
                key=lambda move: _move_score(self.game_state, move),
                reverse=True,
            )
        return moves


class MCTSAgent:
    """
    MCTS 智能体。

    属性：
        num_rounds: 每次决策的模拟轮数
        temperature: 温度参数（控制探索程度）
    """

    def __init__(self, num_rounds=1000, temperature=1.0):
        self.num_rounds = num_rounds
        self.temperature = temperature
        self._root_player = Player.black
        self._rng = random.Random()

    def select_move(self, game_state: GameState) -> Move:
        """
        为当前局面选择最佳棋步。

        流程：
            1. 创建根节点
            2. 进行 num_rounds 轮模拟：
               a. Selection: 用 UCT 选择路径到叶节点
               b. Expansion: 展开叶节点
               c. Simulation: 随机模拟至终局
               d. Backup: 反向传播结果
            3. 选择访问次数最多的棋步

        Args:
            game_state: 当前游戏状态

        Returns:
            选定的棋步
        """
        self._root_player = game_state.next_player
        root = MCTSNode(game_state)

        if root.is_terminal():
            return Move.pass_turn()

        for _ in range(self.num_rounds):
            node = root

            while not node.is_terminal() and not node.untried_moves and node.children:
                node = node.best_child()

            if not node.is_terminal():
                expanded = node.expand()
                if expanded is not None:
                    node = expanded

            result = self._simulate(node.game_state)
            node.backup(result)

        return self._select_best_move(root)

    def _simulate(self, game_state):
        """
        快速模拟（Rollout）：随机走子至终局。

        【第二小问要求】
        标准 MCTS 使用完全随机走子，但需要实现至少两种优化方法：
        1. 启发式走子策略（如：优先选有气、不自杀、提子的走法）
        2. 限制模拟深度（如：最多走 20-30 步后停止评估）
        3. 其他：快速走子评估（RAVE）、池势启发等

        Args:
            game_state: 起始局面

        Returns:
            从当前玩家视角的结果（1=胜, 0=负, 0.5=和）
        """
        rollout_state = game_state
        max_depth = 24
        depth = 0

        while not rollout_state.is_over() and depth < max_depth:
            move = self._pick_rollout_move(rollout_state)
            rollout_state = rollout_state.apply_move(move)
            depth += 1

        if rollout_state.is_over():
            winner = rollout_state.winner()
            if winner is None:
                return 0.5
            return 1.0 if winner == self._root_player else 0.0

        return self._heuristic_value(rollout_state)

    def _select_best_move(self, root):
        """
        根据访问次数选择最佳棋步。

        Args:
            root: MCTS 树根节点

        Returns:
            最佳棋步
        """
        if not root.children:
            return Move.pass_turn()

        playable_children = [child for child in root.children if child.move is not None and child.move.is_play]
        candidates = playable_children or root.children

        best_visits = max(child.visit_count for child in candidates)
        best_children = [child for child in candidates if child.visit_count == best_visits]
        best_children.sort(key=lambda child: child.value)
        return best_children[0].move or Move.pass_turn()

    def _pick_rollout_move(self, game_state):
        candidates = [move for move in game_state.legal_moves() if not move.is_resign]
        playable = [move for move in candidates if move.is_play]
        if not candidates:
            return Move.pass_turn()

        if playable:
            scored = sorted(
                playable,
                key=lambda move: _move_score(game_state, move),
                reverse=True,
            )
            top_k = scored[: min(4, len(scored))]
            return self._rng.choice(top_k)

        return Move.pass_turn()

    def _heuristic_value(self, game_state):
        result = compute_game_result(game_state)
        board_area = game_state.board.num_rows * game_state.board.num_cols
        score = (result.b - (result.w + result.komi)) / max(1, board_area)
        if self._root_player == Player.white:
            score = -score
        value = 0.5 + score / 2.0
        return max(0.0, min(1.0, value))


def _move_score(game_state, move):
    if move.is_pass:
        return -0.25
    if move.is_resign:
        return -100.0

    next_state = game_state.apply_move(move)
    board = game_state.board
    next_board = next_state.board
    player = game_state.next_player
    opponent = player.other

    score = 0.0
    center_row = (board.num_rows + 1) / 2.0
    center_col = (board.num_cols + 1) / 2.0
    score -= abs(move.point.row - center_row) + abs(move.point.col - center_col)

    before_opponent = _count_stones(board, opponent)
    after_opponent = _count_stones(next_board, opponent)
    score += 4.0 * (before_opponent - after_opponent)

    string = next_board.get_go_string(move.point)
    if string is not None:
        score += 0.4 * string.num_liberties

    for neighbor in move.point.neighbors():
        if not board.is_on_grid(neighbor):
            continue
        neighbor_string = next_board.get_go_string(neighbor)
        if neighbor_string is None or neighbor_string.color != opponent:
            continue
        if neighbor_string.num_liberties == 1:
            score += 1.5

    return score


def _count_stones(board, player):
    count = 0
    for row in range(1, board.num_rows + 1):
        for col in range(1, board.num_cols + 1):
            if board.get(Point(row, col)) == player:
                count += 1
    return count
