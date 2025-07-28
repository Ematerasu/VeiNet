import argparse

from scripts_of_tribute.game import Game

from AI.vei import Vei


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--enemy", required=True)
    ap.add_argument("--games", required=True)
    ap.add_argument("--timeout", default=10)
    args = ap.parse_args()

    vei = Vei(f"Vei", weights=args.weights)
    g = Game()
    g.register_bot(vei)
    print(f'{vei.bot_name} vs {args.enemy}')
    g.run(vei.bot_name, args.enemy, runs=args.games, threads=4, timeout=args.timeout, start_game_runner=True, base_client_port=30000, base_server_port=30050, verbose=True)
    print(f'{args.enemy} vs {vei.bot_name}')
    g.run(args.enemy, vei.bot_name, runs=args.games, threads=4, timeout=args.timeout, start_game_runner=True, base_client_port=30000, base_server_port=30050, verbose=True)

if __name__ == "__main__":
    main()