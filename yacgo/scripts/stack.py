"""
Runs the entire yacgo stack: 1 Trainer, 1 DataBroker, n InferenceServers, k Gameplay clients.
"""

import atexit
import multiprocessing as mp
import os
import signal
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process

import randomname

import wandb
from yacgo.data import DataBroker
from yacgo.eval import ModelCompetition
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


def competition_worker(model, best_model, return_val, args):
    """Wrapper around a simple competition worker.

    Args:
        port (int): Port server is listening on.
        args (dict): args dict.
    """
    # run the competition
    servers = []

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
                args=(BEST_MODEL_PORT, args, best_model),
                daemon=True,
            ),
            Process(
                target=inference_worker,
                args=(CANDIDATE_PORT, args, model),
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

    return_val.value = result.probs[0]
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
    best_model = args.model_path
    if best_model is not None:
        epoch = model_name_to_epoch(best_model) + 1
    else:
        epoch = args.epoch
    print("Starting training loop...")
    servers = []

    def close():
        for server in servers:
            if server is not None:
                server.terminate()
                server.join()

    atexit.register(close)
    manager = mp.Manager()
    while True:
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
        candidate = trainer.save_pretrained(candidate=True)

        for server in servers:
            server.terminate()
            server.join()

        servers = []  # gc the old servers
        return_val = manager.Value(float, 0.0)
        result = competition_worker(candidate, best_model, return_val, args)
        if args.wandb:
            wandb.log(
                {
                    "score": result.score,
                    "new_model_win%": result.probs[0],
                    "old_model_win%": result.probs[1],
                    "epoch": epoch,
                }
            )
        if os.path.exists(candidate):
            os.remove(candidate)

        if return_val.value >= 0.55:
            best_model = trainer.save_pretrained(epoch=epoch)
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
