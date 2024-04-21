"""
Runs the entire yacgo stack: 1 Trainer, 1 DataBroker, n InferenceServers, k Gameplay clients.
"""

import atexit
from datetime import datetime as dt
import multiprocessing as mp
import os
from typing import List
import signal
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process

import randomname

import wandb
from yacgo.data import DataBroker
from yacgo.eval import ModelCompetition, CompetitionResult
from yacgo.elo import GameRecord, GameResultSummary
from yacgo.models import InferenceClient, InferenceServer, Trainer
from yacgo.train_utils import GameGenerator
from yacgo.utils import make_args, model_name_to_epoch

BEST_MODEL_PORT = 31337
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
    model: str, best_model: str = None, args: dict = None
) -> CompetitionResult:
    """Wrapper around a simple competition worker.

    Args:
        port (int): Port server is listening on.
        args (dict): args dict.
    """
    # run the competition
    servers = []

    if best_model == "random":
        best_model = None

    def close():
        for server in servers:
            if server is not None:
                server.terminate()
                server.join()

    atexit.register(close)

    if best_model is not None:
        servers = [
            Process(
                target=inference_worker,
                args=(CANDIDATE_PORT, args, model),
                daemon=True,
            ),
            Process(
                target=inference_worker,
                args=(BEST_MODEL_PORT, args, best_model),
                daemon=True,
            ),
        ]
    else:
        servers = [
            Process(
                target=inference_worker,
                args=(CANDIDATE_PORT, args, model),
                daemon=True,
            ),
        ]
    for server in servers:
        server.start()
    if best_model is not None:
        comp = ModelCompetition(CANDIDATE_PORT, BEST_MODEL_PORT, args)
    else:
        comp = ModelCompetition(CANDIDATE_PORT, None, args)
    result = comp.compete(num_games=args.num_comp_games)

    for server in servers:
        server.terminate()
        server.join()

    return result


def trainer_worker(ports, args):
    """Wrapper around a simple trainer worker.

    Args:
        args (dict): args dict.
    """

    trainer = Trainer(args)
    model = None
    best_model = args.model_path
    models = []
    if args.epoch == 0 and best_model is not None:
        epoch = model_name_to_epoch(best_model) + 1
        best_model_name = f"model-{dt.now().strftime('%Y%m%d-%H%M%S')}-{epoch - 1:03d}"
        models.append((best_model_name, best_model))
    else:
        epoch = 0
        best_model_name = "random"
    epoch = max(args.epoch, epoch)

    # Do an initial competition if no history. TODO: create a saveable history
    if best_model is not None and args.starting_elo == 0.0:
        summary = GameResultSummary(
            elo_prior_games=1.0,
            estimate_first_player_advantage=False,
            prior_player=("random", 0),
        )
        print("Performing initial competitions to determine starting ELO")

        result = competition_worker(best_model, None, args)
        win, loss = result.wins_losses
        draw = args.num_comp_games - win - loss
        summary.add_game_record(
            GameRecord(best_model_name, "random", win=win, loss=loss, draw=draw)
        )
        elos = summary.get_elos(recompute=True)

        args.starting_elo = elos.get_elo(best_model_name)
    else:
        summary = GameResultSummary(
            elo_prior_games=1.0,
            estimate_first_player_advantage=False,
            prior_player=(best_model_name, args.starting_elo),
        )

    wandb.log(
        {
            "global_step": trainer.global_step,
            "elo": args.starting_elo,
        }
    )
    print("Starting training loop...")
    servers = []

    def close():
        for server in servers:
            if server is not None:
                server.terminate()
                server.join()

    atexit.register(close)

    while True:
        for port in ports:
            servers.append(
                Process(
                    target=inference_worker,
                    args=(port, args, model),
                    daemon=True,
                )
            )

        for server in servers:
            server.start()

        # run a training run and save the model
        trainer.run(epoch)
        candidate = trainer.save_pretrained(candidate=True, path=args.models_dir)

        for server in servers:
            server.terminate()
            server.join()
        model_name = f"model-{dt.now().strftime('%Y%m%d-%H%M%S')}-{epoch:03d}"
        servers = []  # gc the old servers

        result = competition_worker(candidate, best_model, args)

        win, loss = result.wins_losses
        draw = args.num_comp_games - win - loss

        summary.add_game_record(
            GameRecord(model_name, best_model_name, win=win, loss=loss, draw=draw)
        )

        elos = summary.get_elos(recompute=True)
        print("Raw result", result.probs)
        print("Competition Results")
        print("=" * 80)
        summary.print_game_results()
        print("-" * 80)
        print(elos, "\n")

        if args.wandb:
            wandb.log(
                {
                    "score": result.score,
                    "elo": elos.get_elo(model_name),
                    "new_model_win%": result.probs[0],
                    "epoch": epoch,
                    "global_step": trainer.global_step,
                }
            )
        if os.path.exists(candidate):
            os.remove(candidate)

        model = trainer.save_pretrained(epoch=epoch)
        if result.probs[0] >= args.acceptance_ratio:
            best_model = model
            best_model_name = model_name
        epoch += 1


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
