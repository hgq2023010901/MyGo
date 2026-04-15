"""围棋 AI 对弈 GUI。

功能：
- 支持黑白双方分别选择 human / random / mcts / minimax
- 支持人机对弈、机机对弈
- 支持开始、暂停、单步执行、重开、悔棋、过手、认输
- 显示棋盘、当前回合、提子数、最近一步与对局结果
"""

from __future__ import annotations

import concurrent.futures
import tkinter as tk
from tkinter import ttk

from agents.mcts_agent import MCTSAgent
from agents.minimax_agent import MinimaxAgent
from agents.random_agent import RandomAgent
from dlgo import GameState, Player, Point
from dlgo.goboard import Move


AGENT_OPTIONS = ("human", "random", "mcts", "minimax")


class GoAIGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("围棋 AI 对弈 GUI")

        self.board_size_var = tk.IntVar(value=5)
        self.black_agent_var = tk.StringVar(value="human")
        self.white_agent_var = tk.StringVar(value="mcts")
        self.mcts_rounds_var = tk.IntVar(value=100)
        self.minimax_depth_var = tk.IntVar(value=2)
        self.move_delay_var = tk.IntVar(value=180)

        self.status_var = tk.StringVar(value="准备开始")
        self.turn_var = tk.StringVar(value="当前回合: 黑")
        self.capture_var = tk.StringVar(value="提子  黑:0  白:0")
        self.last_move_var = tk.StringVar(value="最近一步: -")

        self.game_state = GameState.new_game(self.board_size_var.get())
        self.move_count = 0
        self.running = False
        self.game_started = False
        self.black_captures = 0
        self.white_captures = 0
        self.history = []

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.pending_future: concurrent.futures.Future | None = None

        self.board_margin = 28
        self.cell_size = 60
        self.canvas_size = self.board_margin * 2 + self.cell_size * (self.board_size_var.get() - 1)

        self._build_ui()
        self._new_game()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        control = ttk.LabelFrame(main, text="对局设置", padding=10)
        control.grid(row=0, column=0, sticky="ew")

        ttk.Label(control, text="棋盘大小").grid(row=0, column=0, sticky="w")
        size_box = ttk.Combobox(control, width=6, textvariable=self.board_size_var, state="readonly")
        size_box["values"] = (5, 7, 9)
        size_box.grid(row=0, column=1, padx=(6, 12), sticky="w")

        ttk.Label(control, text="黑方").grid(row=0, column=2, sticky="w")
        black_box = ttk.Combobox(control, width=10, textvariable=self.black_agent_var, state="readonly")
        black_box["values"] = AGENT_OPTIONS
        black_box.grid(row=0, column=3, padx=(6, 12), sticky="w")

        ttk.Label(control, text="白方").grid(row=0, column=4, sticky="w")
        white_box = ttk.Combobox(control, width=10, textvariable=self.white_agent_var, state="readonly")
        white_box["values"] = AGENT_OPTIONS
        white_box.grid(row=0, column=5, padx=(6, 12), sticky="w")

        ttk.Label(control, text="MCTS轮数").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(control, width=8, textvariable=self.mcts_rounds_var).grid(
            row=1, column=1, padx=(6, 12), sticky="w", pady=(8, 0)
        )

        ttk.Label(control, text="Minimax深度").grid(row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(control, width=8, textvariable=self.minimax_depth_var).grid(
            row=1, column=3, padx=(6, 12), sticky="w", pady=(8, 0)
        )

        ttk.Label(control, text="步间隔(ms)").grid(row=1, column=4, sticky="w", pady=(8, 0))
        ttk.Entry(control, width=8, textvariable=self.move_delay_var).grid(
            row=1, column=5, padx=(6, 12), sticky="w", pady=(8, 0)
        )

        button_bar = ttk.Frame(main)
        button_bar.grid(row=1, column=0, sticky="ew", pady=(10, 8))

        ttk.Button(button_bar, text="开始", command=self._start).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(button_bar, text="暂停", command=self._pause).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(button_bar, text="单步", command=self._step_once).grid(row=0, column=2, padx=(0, 8))
        ttk.Button(button_bar, text="重开", command=self._new_game).grid(row=0, column=3, padx=(0, 8))
        ttk.Button(button_bar, text="悔棋", command=self._undo).grid(row=0, column=4, padx=(0, 8))
        ttk.Button(button_bar, text="过手", command=self._human_pass).grid(row=0, column=5, padx=(0, 8))
        ttk.Button(button_bar, text="认输", command=self._human_resign).grid(row=0, column=6)

        self.canvas_frame = ttk.Frame(main)
        self.canvas_frame.grid(row=2, column=0, sticky="nsew")
        main.rowconfigure(2, weight=1)

        self.canvas = tk.Canvas(
            self.canvas_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="#d7a66d",
            highlightthickness=1,
            highlightbackground="#9b6f3a",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        info = ttk.Frame(main)
        info.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(info, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        ttk.Label(info, textvariable=self.turn_var).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(info, textvariable=self.capture_var).grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Label(info, textvariable=self.last_move_var).grid(row=3, column=0, sticky="w", pady=(4, 0))

    def _new_game(self) -> None:
        self.running = False
        self.game_started = False
        self.pending_future = None

        size = self.board_size_var.get()
        self.game_state = GameState.new_game(size)
        self.move_count = 0
        self.black_captures = 0
        self.white_captures = 0
        self.history = []

        self.canvas_size = self.board_margin * 2 + self.cell_size * (size - 1)
        self.canvas.config(width=self.canvas_size, height=self.canvas_size)

        self.status_var.set("新对局已创建")
        self.last_move_var.set("最近一步: -")
        self._update_info_panel()
        self._redraw_board()

    def _start(self) -> None:
        if self.game_state.is_over():
            self.status_var.set("对局已结束，请重开")
            return
        self.running = True
        self.game_started = True
        self.status_var.set("对弈进行中")
        self._schedule_next_move()

    def _pause(self) -> None:
        self.running = False
        self.status_var.set("已暂停")

    def _step_once(self) -> None:
        if self.pending_future is not None:
            return
        if self.game_state.is_over():
            self.status_var.set("对局已结束，请重开")
            return
        self.running = False
        self.game_started = True
        self.status_var.set("单步执行")
        self._schedule_next_move(force_single_step=True)

    def _undo(self) -> None:
        if self.pending_future is not None:
            self.status_var.set("请等待当前AI计算完成")
            return
        if not self.history:
            self.status_var.set("无法悔棋：当前没有可撤销步")
            return

        snap = self.history.pop()
        self.game_state = snap["state"]
        self.move_count = snap["move_count"]
        self.black_captures = snap["black_captures"]
        self.white_captures = snap["white_captures"]
        self.last_move_var.set(snap["last_move_text"])
        self.running = False
        self.status_var.set("已悔棋")
        self._update_info_panel()
        self._redraw_board()

    def _human_pass(self) -> None:
        if not self.game_started:
            self.status_var.set("请先点击开始")
            return
        if not self._is_human_turn():
            self.status_var.set("当前不是人类方回合")
            return
        self._apply_move(Move.pass_turn())
        if self.running and not self.game_state.is_over():
            self.root.after(max(1, self.move_delay_var.get()), self._schedule_next_move)

    def _human_resign(self) -> None:
        if not self.game_started:
            self.status_var.set("请先点击开始")
            return
        if not self._is_human_turn():
            self.status_var.set("当前不是人类方回合")
            return
        self._apply_move(Move.resign())

    def _schedule_next_move(self, force_single_step: bool = False) -> None:
        if self.pending_future is not None:
            return
        if self.game_state.is_over():
            self._finish_game()
            return
        if not self.running and not force_single_step:
            return

        next_player = self.game_state.next_player
        agent_name = self.black_agent_var.get() if next_player == Player.black else self.white_agent_var.get()

        if agent_name == "human":
            self.status_var.set("等待人类落子（点击棋盘）")
            return

        state_snapshot = self.game_state
        self.pending_future = self.executor.submit(
            self._compute_move,
            agent_name,
            state_snapshot,
            self.mcts_rounds_var.get(),
            self.minimax_depth_var.get(),
        )
        self.root.after(10, lambda: self._poll_pending(force_single_step))

    def _poll_pending(self, force_single_step: bool) -> None:
        if self.pending_future is None:
            return

        if not self.pending_future.done():
            self.root.after(15, lambda: self._poll_pending(force_single_step))
            return

        future = self.pending_future
        self.pending_future = None

        try:
            move = future.result()
        except Exception as exc:  # noqa: BLE001
            self.running = False
            self.status_var.set(f"落子失败: {exc}")
            return

        if move is None:
            move = Move.pass_turn()

        if not self.game_state.is_valid_move(move):
            legal = [m for m in self.game_state.legal_moves() if not m.is_resign]
            move = legal[0] if legal else Move.pass_turn()

        self._apply_move(move)

        if not self.game_state.is_over() and self.running and not force_single_step:
            delay = max(1, self.move_delay_var.get())
            self.root.after(delay, self._schedule_next_move)
        elif not self.game_state.is_over() and force_single_step:
            self.status_var.set("单步完成")

    @staticmethod
    def _compute_move(agent_name: str, game_state: GameState, mcts_rounds: int, minimax_depth: int) -> Move:
        if agent_name == "random":
            agent = RandomAgent()
        elif agent_name == "mcts":
            agent = MCTSAgent(num_rounds=max(1, int(mcts_rounds)))
        elif agent_name == "minimax":
            agent = MinimaxAgent(
                max_depth=max(1, int(minimax_depth)),
                time_limit=2.0,
                max_branching=12,
            )
        else:
            return Move.pass_turn()
        return agent.select_move(game_state)

    def _apply_move(self, move: Move) -> None:
        if not self.game_state.is_valid_move(move):
            self.status_var.set("非法落子，请重试")
            return

        player = self.game_state.next_player
        self.history.append(
            {
                "state": self.game_state,
                "move_count": self.move_count,
                "black_captures": self.black_captures,
                "white_captures": self.white_captures,
                "last_move_text": self.last_move_var.get(),
            }
        )

        before_opp = self._count_stones(self.game_state.board, player.other)
        next_state = self.game_state.apply_move(move)
        after_opp = self._count_stones(next_state.board, player.other)
        captured = max(0, before_opp - after_opp) if move.is_play else 0

        if player == Player.black:
            self.black_captures += captured
        else:
            self.white_captures += captured

        self.game_state = next_state
        self.move_count += 1

        player_name = "黑" if player == Player.black else "白"
        self.last_move_var.set(f"最近一步: 第{self.move_count}手 {player_name} -> {move}")
        self._update_info_panel()
        self._redraw_board()

        if self.game_state.is_over():
            self._finish_game()
        elif not self.running:
            self.status_var.set("等待下一步")

    def _finish_game(self) -> None:
        self.running = False
        winner = self.game_state.winner()
        if winner is None:
            self.status_var.set("对局结束: 平局")
        else:
            self.status_var.set(f"对局结束: {'黑方' if winner == Player.black else '白方'}获胜")
        self._update_info_panel()

    def _update_info_panel(self) -> None:
        current = "黑" if self.game_state.next_player == Player.black else "白"
        self.turn_var.set(f"当前回合: {current}  |  手数: {self.move_count}")
        self.capture_var.set(f"提子  黑:{self.black_captures}  白:{self.white_captures}")

    def _is_human_turn(self) -> bool:
        next_player = self.game_state.next_player
        next_agent = self.black_agent_var.get() if next_player == Player.black else self.white_agent_var.get()
        return next_agent == "human"

    def _on_canvas_click(self, event: tk.Event) -> None:
        if self.pending_future is not None:
            return
        if not self.game_started:
            self.status_var.set("请先点击开始")
            return
        if self.game_state.is_over():
            self.status_var.set("对局已结束，请重开")
            return
        if not self._is_human_turn():
            self.status_var.set("当前由AI落子")
            return

        point = self._xy_to_point(event.x, event.y)
        if point is None:
            self.status_var.set("请点击棋盘交叉点附近")
            return

        move = Move.play(point)
        if not self.game_state.is_valid_move(move):
            self.status_var.set("该点不可落子")
            return

        self._apply_move(move)
        if self.running and not self.game_state.is_over():
            delay = max(1, self.move_delay_var.get())
            self.root.after(delay, self._schedule_next_move)

    def _redraw_board(self) -> None:
        self.canvas.delete("all")

        size = self.game_state.board.num_rows
        margin = self.board_margin
        span = self.cell_size * (size - 1)

        for i in range(size):
            x = margin + i * self.cell_size
            self.canvas.create_line(x, margin, x, margin + span, fill="#333", width=1)

        for i in range(size):
            y = margin + i * self.cell_size
            self.canvas.create_line(margin, y, margin + span, y, fill="#333", width=1)

        star_points = self._star_points(size)
        for r, c in star_points:
            x, y = self._point_to_xy(Point(r, c))
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#222", outline="#222")

        for row in range(1, size + 1):
            for col in range(1, size + 1):
                stone = self.game_state.board.get(Point(row, col))
                if stone is None:
                    continue
                x, y = self._point_to_xy(Point(row, col))
                radius = 18
                if stone == Player.black:
                    self.canvas.create_oval(
                        x - radius,
                        y - radius,
                        x + radius,
                        y + radius,
                        fill="#111",
                        outline="#000",
                        width=1,
                    )
                else:
                    self.canvas.create_oval(
                        x - radius,
                        y - radius,
                        x + radius,
                        y + radius,
                        fill="#f1f1f1",
                        outline="#888",
                        width=1,
                    )

        for i in range(size):
            text = str(i + 1)
            x = margin + i * self.cell_size
            y = margin + i * self.cell_size
            self.canvas.create_text(x, margin - 14, text=text, fill="#2b2b2b")
            self.canvas.create_text(margin - 14, y, text=text, fill="#2b2b2b")

    def _point_to_xy(self, point: Point) -> tuple[int, int]:
        x = self.board_margin + (point.col - 1) * self.cell_size
        y = self.board_margin + (point.row - 1) * self.cell_size
        return x, y

    def _xy_to_point(self, x: int, y: int) -> Point | None:
        row = round((y - self.board_margin) / self.cell_size) + 1
        col = round((x - self.board_margin) / self.cell_size) + 1
        candidate = Point(row=row, col=col)

        if not self.game_state.board.is_on_grid(candidate):
            return None

        px, py = self._point_to_xy(candidate)
        tolerance = self.cell_size * 0.38
        if abs(px - x) > tolerance or abs(py - y) > tolerance:
            return None
        return candidate

    @staticmethod
    def _count_stones(board, player: Player) -> int:
        count = 0
        for row in range(1, board.num_rows + 1):
            for col in range(1, board.num_cols + 1):
                if board.get(Point(row, col)) == player:
                    count += 1
        return count

    @staticmethod
    def _star_points(size: int) -> list[tuple[int, int]]:
        if size < 7:
            mid = (size + 1) // 2
            return [(mid, mid)]
        if size == 7:
            return [(2, 2), (2, 6), (6, 2), (6, 6), (4, 4)]
        if size >= 9:
            return [(3, 3), (3, 7), (7, 3), (7, 7), (5, 5)]
        return []

    def _on_close(self) -> None:
        self.running = False
        if self.pending_future is not None:
            self.pending_future.cancel()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    GoAIGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
