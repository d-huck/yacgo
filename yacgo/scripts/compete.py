import signal
from multiprocessing import Process
from concurrent.futures import ProcessPoolExecutor as Pool
import multiprocessing as mp
import atexit
from itertools import combinations
from tqdm.auto import tqdm

from yacgo.models import InferenceServer
from yacgo.elo import GameRecord, GameResultSummary
from yacgo.eval import ModelCompetition
from yacgo.utils import make_args
import os
import random

CANDIDATE_PORT = 31338


def inference_worker(port, args, model):
    """Wrapper around a simple inference worker.

    Args:
        port (int): Port server is listening on.
        args (dict): args dict.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    server = InferenceServer(port, args, model)
    server.run()


def competition_worker(inputs):
    names, ports, args = inputs
    p1, p2 = ports
    comp = ModelCompetition(p1, p2, args)

    result = comp.compete(num_games=args.num_comp_games)
    win, loss = result.wins_losses
    draw = args.num_comp_games - win - loss

    m1, m2 = names
    game_record = GameRecord(m1, m2, win=win, loss=loss, draw=draw)
    return game_record


def main():
    mp.set_start_method("spawn")
    args = make_args()
    models = ["random"] + [
        os.path.join(args.model_path, x)
        for x in os.listdir(args.model_path)
        if x.endswith(".pth")
    ]
    games = set([x for x in combinations(models, 2)])
    print("Games:", len(games))
    rand_player = set([x for x in games if "random" in x])
    pbar = tqdm(total=len(games))
    summary = GameResultSummary(
        elo_prior_games=1.0,
        estimate_first_player_advantage=False,
        prior_player=("random", 0.00),
    )
    while len(games) > 0:
        matches = []
        try:
            for _ in range(4):
                matches.append(games.pop())
        except KeyError:
            print("oof")

        m = set([x[0] for x in matches] + [x[1] for x in matches])
        # print(len(m), m)
        matches = matches + [x for x in combinations(m, 2)]
        games = games - set(matches)
        print(
            "Games:", len(games), "Current:", len(matches), "between", len(m), "players"
        )
        servers = []
        ports = {}
        for i, model in enumerate(m):
            port = args.inference_server_port + i
            if model == "random":
                ports[model] = None
            else:
                ports[model] = port
                servers.append(
                    Process(
                        target=inference_worker,
                        args=(port, args, model),
                        daemon=True,
                    )
                )
        for server in servers:
            server.start()

        with Pool(max_workers=4) as pool:
            for results in pool.map(
                competition_worker,
                [(x, [ports[x[0]], ports[x[1]]], args) for x in matches],
            ):
                summary.add_game_record(results)
                elos = summary.get_elos(recompute=True)
                print(elos)

        for server in servers:
            server.terminate()
            server.join()

    elos = summary.get_elos(recompute=True)
    summary.to_csv("final_results.csv")

    # server = Process(target=inference_worker, args=(CANDIDATE_PORT, args), daemon=True)
    # server.start()

    # comp = ModelCompetition(CANDIDATE_PORT, None, args)
    # result = comp.compete(100)

    # print(result)
    # server.terminate()
    # server.join()


if __name__ == "__main__":
    main()
