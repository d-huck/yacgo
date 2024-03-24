"""
Runs the entire yacgo stack: 1 Trainer, 1 DataBroker, n InferenceServers, k Gameplay clients.
"""

import os
import time
import multiprocessing as mp
from multiprocessing import Process

from yacgo.data import DataBroker, DataGameClientMixin
from yacgo.eval import ModelCompetition
from yacgo.models import InferenceClient, InferenceServer, Trainer
from yacgo.train_utils import GameGenerator
from yacgo.utils import make_args

OLD_MODEL_PORT = 31337
NEW_MODEL_PORT = 31338


def inference_worker(port, args, model):
    """Wrapper around a simple inference worker.

    Args:
        port (int): Port server is listening on.
        args (dict): args dict.
    """
    server = InferenceServer(port, args, model)
    server.run()


def databroker_worker(args):
    """Wrapper around a simple databroker worker.

    Args:
        port (int): Port server is listening on.
        args (dict): args dict.
    """
    broker = DataBroker(args)
    broker.run()


def gameplay_worker(ports, i, display, args):
    """Wrapper around a simple gameplay worker.

    Args:
        port (int): Port server is listening on.
        args (dict): args dict.
    """
    model = InferenceClient(ports, args.inference_server_address)
    game_gen = GameGenerator(model, args, display=display)
    # data = game_gen.sim_data(1024)
    # for d in data:
    #     data_client.deposit(d)
    try:
        while True:
            game_gen.sim_game()
    except KeyboardInterrupt:
        print("Quitting game generation, closing sockets...")
        game_gen.destroy()


def trainer_worker(args):
    """Wrapper around a simple trainer worker.

    Args:
        args (dict): args dict.
    """
    # databroker = Process(target=databroker_worker, args=(args,), daemon=True)
    # databroker.start()
    trainer = Trainer(args)
    old_model = args.model_path
    epoch = args.epoch
    print("Starting training loop...")
    while True:
        try:
            model = None
            servers = []
            games = []
            ports = list(
                range(
                    args.inference_server_port,
                    args.inference_server_port + args.num_servers,
                )
            )
            print("Ports", ports)
            for port in ports:
                servers.append(
                    Process(
                        target=inference_worker,
                        args=(port, args, old_model),
                        daemon=True,
                    )
                )

            for server in servers:
                server.start()
            display = True
            for i in range(args.num_games):
                games.append(
                    Process(
                        target=gameplay_worker,
                        args=(ports, i, display, args),
                        daemon=True,
                    )
                )
                display &= False
            for i, game in enumerate(games):
                game.start()
                if i % 16 == 15:  # slow spin up of the games
                    time.sleep(5)

            # run a training run and save the model
            trainer.run(epoch)
            model = trainer.save_pretrained(iter=epoch)

            # shutdown infererers
            for server in servers:
                server.terminate()
                server.join()

            for game in games:
                game.terminate()
                game.join()

            # run the competition
            if old_model is not None:
                servers = [
                    Process(
                        target=inference_worker,
                        args=(OLD_MODEL_PORT, args, old_model),
                        daemon=True,
                    ),
                    Process(
                        target=inference_worker,
                        args=(NEW_MODEL_PORT, args, model),
                        daemon=True,
                    ),
                ]
            else:
                servers = [
                    Process(
                        target=inference_worker,
                        args=(NEW_MODEL_PORT, args, model),
                        daemon=True,
                    ),
                ]
            for server in servers:
                server.start()
            if old_model is not None:
                comp = ModelCompetition(
                    NEW_MODEL_PORT, OLD_MODEL_PORT, args, n_workers=args.num_games
                )
            else:
                comp = ModelCompetition(
                    NEW_MODEL_PORT, None, args, n_workers=args.num_games
                )
            result = comp.compete(num_games=args.num_comp_games)
            if result.probs[0] >= 0.55:
                old_model = model
                trainer.reset_data()
                iter += 1
            else:
                trainer.load_pretrained(old_model)
                os.remove(model)
            for server in servers:
                server.terminate()
                server.join()
            del comp
        except KeyboardInterrupt:
            print("Quitting training, closing sockets...")
            for server in servers:
                if server is not None:
                    server.terminate()
                    server.join()
            for game in games:
                if game is not None:
                    game.terminate()
                    game.join()
            if model is not None:
                os.remove(model)  # don't store a bad model
            break


def main():
    """
    Main Process
    """
    mp.set_start_method("spawn")
    args = make_args()
    try:
        # Start the trainer
        trainer = Process(target=trainer_worker, args=(args,))
        trainer.start()
        trainer.join()

    except KeyboardInterrupt:
        pass
    finally:
        print("Exiting...")


if __name__ == "__main__":
    main()
