"""
Runs the entire yacgo stack: 1 Trainer, 1 DataBroker, n InferenceServers, k Gameplay clients.
"""

import atexit
import multiprocessing as mp
import os
import signal
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime as dt
from itertools import combinations
from multiprocessing import Process
from typing import List

import randomname
from tqdm.auto import tqdm

import wandb
from yacgo.data import DataBroker
from yacgo.elo import GameRecord, GameResultSummary
from yacgo.eval import CompetitionResult, ModelCompetition
from yacgo.models import InferenceClient, InferenceServer, Trainer
from yacgo.train_utils import GameGenerator
from yacgo.utils import make_args, model_name_to_epoch

MODEL_1_PORT = 31337
MODEL_2_PORT = 31338


def inference_worker(port, args, model):
    """Wrapper around a simple inference worker.

    Args:
        port (int): Port server is listening on.
        args (dict): args dict.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    server = InferenceServer(port, args, model)
    server.run()


def databroker_worker(args):
    """Wrapper around a simple databroker worker.

    Args:
        port (int): Port server is listening on.
        args (dict): args dict.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            job_type="replay_buffer",
            config=args,
        )
    broker = DataBroker(args)
    broker.run()


def gameplay_worker(ports, display, args):
    """Wrapper around a simple gameplay worker.

    Args:
        port (int): Port server is listening on.
        args (dict): args dict.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    def game_play_thread():
        model = InferenceClient(ports, args.inference_server_address)
        game_gen = GameGenerator(model, args, display=display)

        try:
            while True:
                game_gen.sim_game()
        except KeyboardInterrupt:
            print("Quitting game generation, closing sockets...")
            game_gen.destroy()

    n_threads = args.num_games // args.num_game_processes
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        for _ in range(n_threads):
            executor.submit(game_play_thread)


def competition_worker(
    models: list, summary: GameResultSummary, args: dict = None
) -> GameResultSummary:
    """Wrapper around a simple competition worker.

    Args:
        port (int): Port server is listening on.
        args (dict): args dict.
    """
    # run the competition
    servers = []
    records = []

    def close():
        for server in servers:
            if server is not None:
                server.terminate()
                server.join()

    atexit.register(close)

    for model1, model2 in tqdm(
        combinations(models, 2),
        position=0,
        leave=False,
        total=len(models) * (len(models) - 1) // 2,
    ):
        if model1 == "random":
            model1 = None
        elif model2 == "random":
            model2 = None

        servers = []
        if model1 is not None:
            servers.append(
                Process(
                    target=inference_worker,
                    args=(MODEL_1_PORT, args, model1),
                    daemon=True,
                )
            )
        if model2 is not None:
            servers.append(
                Process(
                    target=inference_worker,
                    args=(MODEL_2_PORT, args, model2),
                    daemon=True,
                )
            )

        for server in servers:
            server.start()

        p1 = MODEL_1_PORT if model1 is not None else None
        p2 = MODEL_2_PORT if model2 is not None else None
        comp = ModelCompetition(p1, p2, args)

        result = comp.compete(num_games=args.num_comp_games)
        win, loss = result.wins_losses
        draw = args.num_comp_games - win - loss

        if model1 is None:
            model1 = "random"
        if model2 is None:
            model2 = "random"
        game_record = GameRecord(model1, model2, win=win, loss=loss, draw=draw)
        summary.add_game_record(game_record)

        for server in servers:
            server.terminate()
            server.join()

    elos = summary.get_elos(recompute=True)

    # print("Competition Results")
    # print("=" * 80)
    # summary.print_game_results()
    # print("-" * 80)
    # print(elos, "\n")

    return summary


def trainer_worker(ports, args):
    """Wrapper around a simple trainer worker.

    Args:
        args (dict): args dict.
    """

    trainer = Trainer(args)
    # model = None
    # best_model = args.model_path
    # models = []
    # if args.epoch == 0 and best_model is not None:
    #     epoch = model_name_to_epoch(best_model) + 1
    #     best_model_name = f"model-{dt.now().strftime('%Y%m%d-%H%M%S')}-{epoch - 1:03d}"
    #     models.append((best_model_name, best_model))
    # else:
    #     epoch = 0
    #     best_model_name = "random"
    # epoch = max(args.epoch, epoch)

    if os.path.exists(args.game_records):
        summary = GameResultSummary.from_csv(args.game_records)
        elos = summary.get_elos(recompute=True)
        print("Found Game Records! Current ELOs:")
        print(elos)
        models = elos.get_players()[: args.top_k]
    else:
        summary = GameResultSummary(
            elo_prior_games=1000.0,
            estimate_first_player_advantage=False,
            prior_player=("random", 0),
        )
        models = ["random"]

    servers = []

    def close():
        summary.to_csv(args.game_records)
        for server in servers:
            if server is not None:
                server.terminate()
                server.join()

    atexit.register(close)

    best_model = models[0]
    if best_model != "random":
        trainer.load_pretrained(best_model)
        epoch = model_name_to_epoch(best_model) + 1

    wandb.log(
        {
            "global_step": trainer.global_step,
            "elo": args.starting_elo,
        }
    )
    print("Starting training loop...")

    while True:
        # models = ["random"]
        for _ in range(args.competition_epochs):
            for port in ports:
                servers.append(
                    Process(
                        target=inference_worker,
                        args=(port, args, best_model),
                        daemon=True,
                    )
                )

            for server in servers:
                server.start()

            # run a training run and save the model
            trainer.run(epoch)
            model = trainer.save_pretrained(epoch=epoch, path=args.models_dir)

            for server in servers:
                server.terminate()
                server.join()

            # model_name = f"model-{dt.now().strftime('%Y%m%d-%H%M%S')}-{epoch:03d}"
            models.append(model)
            servers = []  # gc the old servers
            epoch += 1

        summary = competition_worker(models, summary, args)
        elos = summary.get_elos(recompute=True)

        # win, loss = result.wins_losses
        # draw = args.num_comp_games - win - loss

        # elos = summary.get_elos(recompute=True)
        print("Competition Results")
        print("=" * 80)
        summary.print_game_results()
        print("-" * 80)
        print(elos, "\n")

        models = elos.get_players()[: args.top_k]
        best_model = models[0]
        if best_model != "random":
            trainer.load_pretrained(best_model)

        if args.wandb:
            wandb.log(
                {
                    # "score": result.score,
                    "elo": elos.get_elo(models[0]),
                    # "new_model_win%": result.probs[0],
                    "epoch": epoch,
                    "global_step": trainer.global_step,
                }
            )
        # if os.path.exists(candidate):
        #     os.remove(candidate)

        # model = trainer.save_pretrained(epoch=epoch)
        # if result.probs[0] >= args.acceptance_ratio:
        #     best_model = model
        #     best_model_name = model_name


def main():
    """
    Main Process
    """
    mp.set_start_method("spawn")
    args = make_args()
    if args.wandb:
        # wandb.require("service")
        if args.wandb_group is None:
            args.wandb_group = randomname.get_name()
        wandb.init(
            project="yacgo", group=args.wandb_group, job_type="training", config=args
        )
    games = []
    ports = list(
        range(
            args.inference_server_port,
            args.inference_server_port + args.num_servers,
        )
    )
    try:
        # Start the databroker
        databroker = Process(target=databroker_worker, args=(args,), daemon=True)
        databroker.start()

        # Start the trainer
        trainer_worker(ports, args)
        databroker.terminate()
        databroker.join()

    except KeyboardInterrupt:
        pass
    except Exception as e:  # pylint: disable=broad-except
        print("Error:", e)
    finally:
        print("Terminating games...")
        for game in games:
            if game is not None:
                game.terminate()
                game.join()
        if databroker is not None:
            print("Terminating databroker...")
            databroker.terminate()
            databroker.join()

        print("Exiting...")


if __name__ == "__main__":
    main()
