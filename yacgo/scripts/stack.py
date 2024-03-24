"""
Runs the entire yacgo stack: 1 Trainer, 1 DataBroker, n InferenceServers, k Gameplay clients.
"""

import os
from multiprocessing import Process

from yacgo.data import DataBroker, DataGameClientMixin
from yacgo.eval import ModelCompetition
from yacgo.models import InferenceClient, InferenceServer, Trainer
from yacgo.train_utils import GameGenerator
from yacgo.utils import make_args

OLD_MODEL_PORT = 31337
NEW_MODEL_PORT = 31338


def inference_worker(port, args):
    """Wrapper around a simple inference worker.

    Args:
        port (int): Port server is listening on.
        args (dict): args dict.
    """
    server = InferenceServer(port, args)
    server.run()


def databroker_worker(args):
    """Wrapper around a simple databroker worker.

    Args:
        port (int): Port server is listening on.
        args (dict): args dict.
    """
    broker = DataBroker(args)
    broker.run()


def trainer_worker(args):
    """Wrapper around a simple trainer worker.

    Args:
        args (dict): args dict.
    """
    databroker = Process(target=databroker_worker, args=(args,), daemon=True)
    databroker.start()
    trainer = Trainer(args)
    old_model = args.model_path
    iter = args.epoch
    print("Starting training loop...")
    while True:
        servers = []
        games = []
        ports = list(range(args.inference_server_port, args.num_servers))
        for port in ports:
            servers.append(
                Process(
                    target=inference_worker, args=(port, args, old_model), daemon=True
                )
            )

        for server in servers:
            server.start()

        for _ in range(args.num_games):
            games.append(
                Process(target=gameplay_worker, args=(ports, args), daemon=True)
            )
        for game in games:
            game.start()

        # run a training run and save the model
        trainer.run()
        model = trainer.save_pretrained(iter=iter)

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
        comp = ModelCompetition(NEW_MODEL_PORT, OLD_MODEL_PORT, args, n_workers=16)
        result = comp.compete(num_games=400)
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


def gameplay_worker(ports, args):
    """Wrapper around a simple gameplay worker.

    Args:
        port (int): Port server is listening on.
        args (dict): args dict.
    """
    model = InferenceClient(ports)
    data_client = DataGameClientMixin(args)
    game_gen = GameGenerator(
        model,
        args,
    )
    print("Starting Game Generation...")
    data = game_gen.sim_data(1024)
    for d in data:
        data_client.deposit(d)
    while True:
        data = game_gen.sim_game()

        for d in data:
            data_client.deposit(d)


def main():
    """
    Main Process
    """
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
