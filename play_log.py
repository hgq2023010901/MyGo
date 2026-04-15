"""对弈并记录结果到日志文件。

示例：
    python play_log.py --agent1 mcts --agent2 random --size 5 --games 20
    python play_log.py --agent1 minimax --agent2 mcts --size 5 --games 10 --out logs/mvsm.csv
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from dlgo import Player
from play import AGENTS, play_game


def _winner_to_text(winner: Player | None) -> str:
    if winner is None:
        return "draw"
    return "black" if winner == Player.black else "white"


def _default_output_path() -> Path:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("logs") / f"play_log_{now}.csv"


def run_and_log(agent1_name: str, agent2_name: str, size: int, games: int, out_path: Path, quiet: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    agent1_fn = AGENTS[agent1_name]
    agent2_fn = AGENTS[agent2_name]

    results = {Player.black: 0, Player.white: 0, None: 0}
    total_moves = 0
    total_time = 0.0

    fieldnames = [
        "game_index",
        "start_time",
        "end_time",
        "board_size",
        "black_agent",
        "white_agent",
        "winner",
        "moves",
        "duration_seconds",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(1, games + 1):
            start_dt = datetime.now()
            winner, moves, duration = play_game(
                agent1_fn,
                agent2_fn,
                board_size=size,
                verbose=not quiet,
            )
            end_dt = datetime.now()

            writer.writerow(
                {
                    "game_index": i,
                    "start_time": start_dt.isoformat(timespec="seconds"),
                    "end_time": end_dt.isoformat(timespec="seconds"),
                    "board_size": size,
                    "black_agent": agent1_name,
                    "white_agent": agent2_name,
                    "winner": _winner_to_text(winner),
                    "moves": moves,
                    "duration_seconds": f"{duration:.4f}",
                }
            )

            results[winner] += 1
            total_moves += moves
            total_time += duration

    print("\n========== 统计 ==========")
    print(f"对局数: {games}")
    print(f"黑方 ({agent1_name}) 胜: {results[Player.black]}")
    print(f"白方 ({agent2_name}) 胜: {results[Player.white]}")
    print(f"平局: {results[None]}")
    print(f"平均步数: {total_moves / games:.1f}")
    print(f"平均用时: {total_time / games:.2f}s")
    print(f"日志文件: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="围棋 AI 对弈日志记录")
    parser.add_argument("--agent1", choices=AGENTS.keys(), default="random", help="黑方智能体")
    parser.add_argument("--agent2", choices=AGENTS.keys(), default="random", help="白方智能体")
    parser.add_argument("--size", type=int, default=5, help="棋盘大小")
    parser.add_argument("--games", type=int, default=10, help="对局数")
    parser.add_argument("--out", type=str, default="", help="输出日志路径（CSV）")
    parser.add_argument("--quiet", action="store_true", help="静默模式（不打印棋盘）")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else _default_output_path()

    run_and_log(
        agent1_name=args.agent1,
        agent2_name=args.agent2,
        size=args.size,
        games=args.games,
        out_path=out_path,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
