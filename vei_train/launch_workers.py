import csv
import datetime
import re
import shutil
import argparse, subprocess, sys, os, time, signal, pathlib
from typing import List

ROOT = pathlib.Path(__file__).resolve().parents[1]
PYTHON = sys.executable

EVAL_BOTS = [
    "RandomWithoutEndTurnBot",
    "MaxPrestigeBot",
    "DecisionTreeBot",
    "SOISMCTS"
]

RUN_RE   = re.compile(r"Running\s+\d+\s+games\s+-\s+(.+?)\s+vs\s+(.+)")
P1_RE    = re.compile(r"Final amount of P1 wins:\s+(\d+)/(\d+)")
P2_RE    = re.compile(r"Final amount of P2 wins:\s+(\d+)/(\d+)")

def run_eval(weights_path: str, enemy: str, games: int = 100) -> float:
    cmd = [
        PYTHON, "-m", "vei_train.vei_eval",
        f"--weights={weights_path}",
        f"--enemy={enemy}",
        f"--games={games}",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    text = proc.stdout.splitlines()

    total_games = 0
    vei_wins    = 0
    current_side: tuple[str, str] | None = None

    for line in text:
        m = RUN_RE.search(line)
        if m:
            current_side = (m.group(1).strip(), m.group(2).strip())
            continue

        m = P1_RE.search(line)
        if m and current_side:
            p1_wins = int(m.group(1)); games_in_block = int(m.group(2))
            if current_side[0] == "Vei":
                vei_wins += p1_wins
            total_games += games_in_block
            continue

        m = P2_RE.search(line)
        if m and current_side:
            p2_wins = int(m.group(1))
            if current_side[1] == "Vei":
                vei_wins += p2_wins
            continue

    if total_games == 0:
        print(f"[EVAL] Could not parse results vs {enemy}\n"
                f"--- stdout ---\n{proc.stdout}")
        return 0.0
    return vei_wins / total_games

def keep_last_n_checkpoints(ckpt_dir: pathlib.Path, n: int = 5):
    ckpts = sorted(
        ckpt_dir.glob("weights_*.pt"),
        key=lambda p: p.stat().st_mtime
    )
    for old_ckpt in ckpts[:-n]:
        try:
            old_ckpt.unlink()
            print(f"[GC] removed {old_ckpt.name}")
        except Exception as e:
            print(f"[GC] could not remove {old_ckpt.name}: {e}")

def copy_weights(src: str, dst_dir: pathlib.Path, step: int, keep_last: int = 5) -> pathlib.Path:
    dst = dst_dir / f"weights_{step}.pt"
    shutil.copy(src, dst)
    keep_last_n_checkpoints(dst_dir, keep_last)
    return dst

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--weights", default=str(ROOT / "weights.pt"))
    ap.add_argument("--replay-dir", default=str(ROOT / "replay"))
    ap.add_argument("--eval-every",  type=int, default=50)
    ap.add_argument("--games", type=int, default=100)
    args = ap.parse_args()

    os.makedirs(args.replay_dir, exist_ok=True)
    ckpt_dir = pathlib.Path(args.weights).with_suffix("").parent / "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    metrics_csv = ckpt_dir / "metrics.csv"
    first_write = not metrics_csv.exists()

    procs: List[subprocess.Popen] = []
    try:
        for i in range(args.num_workers):
            p = subprocess.Popen(
                [PYTHON, "-m", "vei_train.selfplay_worker",
                    f"--wid={i}",
                    f"--weights={args.weights}",
                    f"--replay-dir={args.replay_dir}"],
                #stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                #text=True, encoding="utf-8"
            )
            procs.append(p)

        last_mtime = pathlib.Path(args.weights).stat().st_mtime if pathlib.Path(args.weights).exists() else 0.0
        checkpoint_idx = 0
        while True:
            if pathlib.Path(args.weights).exists():
                mtime = pathlib.Path(args.weights).stat().st_mtime
                if mtime > last_mtime:
                    checkpoint_idx += 1
                    last_mtime = mtime
                    print(f"\n[MAIN] Detected new checkpoint #{checkpoint_idx}")

                    ckpt_path = copy_weights(args.weights, ckpt_dir, checkpoint_idx, keep_last=10)

                    if checkpoint_idx % args.eval_every == 0 and checkpoint_idx:
                        print(f"[EVAL] running suite for ckpt {checkpoint_idx}")
                        wrs = []
                        for bot in EVAL_BOTS:
                            wr = run_eval(str(ckpt_path), bot, games=args.games)
                            wrs.append(wr)
                            print(f"[EVAL] WR vs {bot}: {wr:.2%}")

                        with open(metrics_csv, "a", newline="") as f:
                            writer = csv.writer(f)
                            if first_write:
                                writer.writerow(["timestamp", "ckpt"] + EVAL_BOTS)
                                first_write = False
                            writer.writerow(
                                [datetime.datetime.now().isoformat(timespec="seconds"), checkpoint_idx] + wrs)

            time.sleep(0.3)

    except KeyboardInterrupt:
        print("\n[MAIN] Ctrl-C – killing workers…")
        for p in procs:
            p.send_signal(signal.SIGINT)
    finally:
        for p in procs:
            p.wait()

if __name__ == "__main__":
    main()
