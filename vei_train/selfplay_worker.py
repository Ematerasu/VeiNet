import random
import itertools
import uuid, json, pathlib, time, os, argparse
from scripts_of_tribute.game import Game
from AI.vei import Vei
import time

BASE_CLIENT_PORT = 50000
BASE_SERVER_PORT = 49000

OPPONENT_PROBS = [
    ("RandomWithoutEndTurnBot", 0.10),
    ("MaxPrestigeBot",          0.20),
    ("DecisionTreeBot",         0.15),
    ("BeamSearchBot",           0.15),
    ("MCTSBot",                 0.05),
    ("SOISMCTS",                0.00),
    ("Vei-mirror",              0.25),   # self‐play
    ("Vei-former",              0.10),
]

def pick_former_weights(current_weights: str) -> str:
    ckpt_dir = pathlib.Path(current_weights).with_suffix("").parent / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("weights_*.pt"))
    return str(random.choice(ckpts)) if ckpts else current_weights

def selfplay_once(weights, replay_dir, worker_id, tag_mtime: float):
    tmp_path = pathlib.Path(replay_dir) / "tmp" / f"{uuid.uuid4().hex}.jsonl"
    
    play_as_first = random.random() < 0.5
    enemy_name_or_class = random.choices(
        [n for n,_ in OPPONENT_PROBS],
        weights=[p for _,p in OPPONENT_PROBS],
        k=1)[0]
    
    vei_bot = Vei(f"Vei_{worker_id}", weights=weights, traj_path=str(tmp_path), tag=tag_mtime)
    if enemy_name_or_class == "Vei-mirror":
        enemy_bot      = Vei(f"VeiEnemy_{worker_id}", weights=weights, traj_path=str(tmp_path), tag=tag_mtime)
        enemy_is_pybot = True
    elif enemy_name_or_class == "Vei-former":
        former_w       = pick_former_weights(weights)
        enemy_bot      = Vei(f"VeiFormer_{worker_id}", weights=former_w)
        enemy_is_pybot = True
    else: 
        enemy_bot      = enemy_name_or_class
        enemy_is_pybot = False

    p1 = vei_bot if play_as_first else enemy_bot
    p2 = enemy_bot if play_as_first else vei_bot

    cport = BASE_CLIENT_PORT + 2 * worker_id
    sport = BASE_SERVER_PORT + 2 * worker_id

    g = Game()
    g.register_bot(vei_bot)
    if enemy_is_pybot:
        g.register_bot(enemy_bot)

    g.run(p1.bot_name if isinstance(p1, Vei) else p1, p2.bot_name if isinstance(p2, Vei) else p2,
        runs=1, threads=1, timeout=20, start_game_runner=True,
        base_client_port=cport, base_server_port=sport, verbose=True)

    final_path = pathlib.Path(replay_dir) / f"{int(time.time()*1e3)}_{worker_id}.jsonl"
    tmp_path.replace(final_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wid", type=int, required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--replay-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.replay_dir, exist_ok=True)
    os.makedirs(args.replay_dir + "/tmp", exist_ok=True)
    last_mtime = 0.0
    while True:
        files = list(pathlib.Path(args.replay_dir).glob("*.jsonl"))
        if len(files) > 100:
            print(f"[W{args.wid}] too many files in {args.replay_dir} ({len(files)}), sleeping 5min...", flush=True)
            time.sleep(300)
            continue

        if os.path.isfile(args.weights):
            mtime = os.path.getmtime(args.weights)
            if mtime > last_mtime:
                print(f"[W{args.wid}] reloading weights ({mtime})", flush=True)
                last_mtime = mtime
            current_tag = last_mtime
        selfplay_once(args.weights if os.path.isfile(args.weights) else None, args.replay_dir, args.wid, tag_mtime=current_tag)
        time.sleep(1)

if __name__ == "__main__":
    main()
