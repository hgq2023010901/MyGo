"""
第三小问（选做）：Minimax 智能体。

实现 Minimax + Alpha-Beta 剪枝算法，并提供一个简单的局面评估函数与
置换表缓存，便于与 MCTS 进行对比。
"""

import math
import time

from dlgo.gotypes import Player, Point
from dlgo.goboard import GameState, Move
from dlgo.scoring import evaluate_territory

__all__ = ["MinimaxAgent"]



class MinimaxAgent:
    """
    Minimax 智能体（带 Alpha-Beta 剪枝）。

    属性：
        max_depth: 搜索最大深度
        evaluator: 局面评估函数
    """

    def __init__(self, max_depth=3, evaluator=None, time_limit=2.0, max_branching=12):
        self.max_depth = max_depth
        self.evaluator = evaluator or self._default_evaluator
        self.time_limit = max(0.05, float(time_limit))
        self.max_branching = max(1, int(max_branching))
        self.cache = GameResultCache()
        self._root_player = Player.black
        self._search_deadline = 0.0

    def select_move(self, game_state: GameState) -> Move:
        """
        为当前局面选择最佳棋步。

        Args:
            game_state: 当前游戏状态

        Returns:
            选定的棋步
        """
        self._root_player = game_state.next_player
        self._search_deadline = time.perf_counter() + self.time_limit

        legal_moves = [move for move in game_state.legal_moves() if not move.is_resign]
        if not legal_moves:
            return Move.pass_turn()

        ordered_root_moves = self._get_ordered_moves(game_state, depth=self.max_depth)
        if not ordered_root_moves:
            return Move.pass_turn()

        best_move = ordered_root_moves[0]

        # 迭代加深：在时间预算内尽量搜索更深，并始终保留可返回的最优着法。
        for depth in range(1, self.max_depth + 1):
            depth_best_move = best_move
            depth_best_value = -math.inf
            try:
                for move in ordered_root_moves:
                    self._check_timeout()
                    next_state = game_state.apply_move(move)
                    value = self.alphabeta(
                        next_state,
                        depth - 1,
                        -math.inf,
                        math.inf,
                        maximizing_player=False,
                    )
                    if value > depth_best_value:
                        depth_best_value = value
                        depth_best_move = move
                best_move = depth_best_move
                ordered_root_moves = self._promote_best_move(ordered_root_moves, best_move)
            except SearchTimeout:
                break

        return best_move

    def minimax(self, game_state, depth, maximizing_player):
        """
        基础 Minimax 算法。

        Args:
            game_state: 当前局面
            depth: 剩余搜索深度
            maximizing_player: 是否在当前层最大化（True=我方）

        Returns:
            该局面的评估值
        """
        if depth <= 0 or game_state.is_over():
            return self.evaluator(game_state)

        moves = self._get_ordered_moves(game_state)
        if not moves:
            return self.evaluator(game_state)

        if maximizing_player:
            value = -math.inf
            for move in moves:
                value = max(
                    value,
                    self.minimax(game_state.apply_move(move), depth - 1, False),
                )
            return value

        value = math.inf
        for move in moves:
            value = min(
                value,
                self.minimax(game_state.apply_move(move), depth - 1, True),
            )
        return value

    def alphabeta(self, game_state, depth, alpha, beta, maximizing_player):
        """
        Alpha-Beta 剪枝优化版 Minimax。

        Args:
            game_state: 当前局面
            depth: 剩余搜索深度
            alpha: 当前最大下界
            beta: 当前最小上界
            maximizing_player: 是否在当前层最大化

        Returns:
            该局面的评估值
        """
        self._check_timeout()

        cache_key = (game_state.next_player, game_state.board.zobrist_hash())
        cached = self.cache.get(cache_key)
        if cached is not None and cached["depth"] >= depth and cached["flag"] == "exact":
            return cached["value"]

        if depth <= 0 or game_state.is_over():
            value = self.evaluator(game_state)
            self.cache.put(cache_key, depth, value, flag="exact")
            return value

        moves = self._get_ordered_moves(game_state, depth=depth)
        if not moves:
            value = self.evaluator(game_state)
            self.cache.put(cache_key, depth, value, flag="exact")
            return value

        if maximizing_player:
            value = -math.inf
            for move in moves:
                value = max(
                    value,
                    self.alphabeta(
                        game_state.apply_move(move),
                        depth - 1,
                        alpha,
                        beta,
                        False,
                    ),
                )
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = math.inf
            for move in moves:
                value = min(
                    value,
                    self.alphabeta(
                        game_state.apply_move(move),
                        depth - 1,
                        alpha,
                        beta,
                        True,
                    ),
                )
                beta = min(beta, value)
                if alpha >= beta:
                    break

        self.cache.put(cache_key, depth, value, flag="exact")
        return value

    def _default_evaluator(self, game_state):
        """
        默认局面评估函数（简单版本）。

        学生作业：替换为更复杂的评估函数，如：
            - 气数统计
            - 眼位识别
            - 神经网络评估

        Args:
            game_state: 游戏状态

        Returns:
            评估值（正数对我方有利）
        """
        if game_state.is_over():
            winner = game_state.winner()
            if winner is None:
                return 0.0
            return 1_000.0 if winner == self._root_player else -1_000.0

        territory = evaluate_territory(game_state.board)
        own = 0.0
        opp = 0.0

        for row in range(1, game_state.board.num_rows + 1):
            for col in range(1, game_state.board.num_cols + 1):
                point = Point(row, col)
                stone = game_state.board.get(point)
                if stone == self._root_player:
                    own += 1.0
                    own += _string_liberties(game_state.board, point) * 0.15
                elif stone == self._root_player.other:
                    opp += 1.0
                    opp += _string_liberties(game_state.board, point) * 0.15

        if self._root_player == Player.black:
            own += territory.num_black_territory
            opp += territory.num_white_territory + 7.5
        else:
            own += territory.num_white_territory + 7.5
            opp += territory.num_black_territory

        return own - opp

    def _get_ordered_moves(self, game_state, depth=0):
        """
        获取排序后的候选棋步（用于优化剪枝效率）。

        好的排序能让 Alpha-Beta 剪掉更多分支。

        Args:
            game_state: 游戏状态

        Returns:
            按启发式排序的棋步列表
        """
        moves = [move for move in game_state.legal_moves() if not move.is_resign]
        ordered = sorted(moves, key=lambda move: _move_score(game_state, move), reverse=True)

        # 深层节点进行分支裁剪，避免 7x7+ 深度搜索时组合爆炸。
        if len(ordered) > self.max_branching and depth >= 2:
            playable = [move for move in ordered if move.is_play]
            trimmed = playable[: self.max_branching]
            if any(move.is_pass for move in ordered):
                trimmed.append(Move.pass_turn())
            return trimmed

        return ordered

    def _check_timeout(self):
        if time.perf_counter() >= self._search_deadline:
            raise SearchTimeout()

    @staticmethod
    def _promote_best_move(moves, best_move):
        if not moves or moves[0] == best_move:
            return moves
        reordered = [best_move]
        reordered.extend(move for move in moves if move != best_move)
        return reordered


class SearchTimeout(Exception):
    """Minimax 搜索超时信号。"""



class GameResultCache:
    """
    局面缓存（Transposition Table）。

    用 Zobrist 哈希缓存已评估的局面，避免重复计算。
    """

    def __init__(self):
        self.cache = {}

    def get(self, zobrist_hash):
        """获取缓存的评估值。"""
        return self.cache.get(zobrist_hash)

    def put(self, zobrist_hash, depth, value, flag='exact'):
        """
        缓存评估结果。

        Args:
            zobrist_hash: 局面哈希
            depth: 搜索深度
            value: 评估值
            flag: 'exact'/'lower'/'upper'（精确值/下界/上界）
        """
        existing = self.cache.get(zobrist_hash)
        if existing is None or depth >= existing["depth"]:
            self.cache[zobrist_hash] = {
                "depth": depth,
                "value": value,
                "flag": flag,
            }


def _move_score(game_state, move):
    if move.is_pass:
        return -0.2
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

    score += 3.0 * (_count_stones(board, opponent) - _count_stones(next_board, opponent))

    string = next_board.get_go_string(move.point)
    if string is not None:
        score += 0.3 * string.num_liberties

    for neighbor in move.point.neighbors():
        if not board.is_on_grid(neighbor):
            continue
        neighbor_string = next_board.get_go_string(neighbor)
        if neighbor_string is None or neighbor_string.color != opponent:
            continue
        if neighbor_string.num_liberties == 1:
            score += 1.2

    return score


def _count_stones(board, player):
    count = 0
    for row in range(1, board.num_rows + 1):
        for col in range(1, board.num_cols + 1):
            if board.get(Point(row, col)) == player:
                count += 1
    return count


def _string_liberties(board, point):
    go_string = board.get_go_string(point)
    if go_string is None:
        return 0
    return go_string.num_liberties
