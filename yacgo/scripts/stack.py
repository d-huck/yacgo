"""
Runs the entire yacgo stack: 1 Trainer, 1 DataBroker, n InferenceServers, k Gameplay clients.
"""

from multiprocessing import Process

from yacgo.data import DataBroker, DataGameClient
from yacgo.models import InferenceClient, InferenceServer, Trainer
from yacgo.train_utils import GameGenerator
from yacgo.utils import make_args


def inference_worker(port, args):
    """Wrapper around a simple inference worker.

    Args:
        port (int): Port server is listening on.
        args (dict): args dict.
    """
    server = InferenceServer(port, args)
    print("Starting server...")
    server.run()


def databroker_worker(args):
    """Wrapper around a simple databroker worker.

    Args:
        port (int): Port server is listening on.
        args (dict): args dict.
    """
    broker = DataBroker(args.databroker_port)
    print("Starting databroker...")
    broker.run()


def trainer_worker(args):
    """Wrapper around a simple trainer worker.

    Args:
        args (dict): args dict.
    """
    trainer = Trainer(args)
    print("Starting trainer...")
    trainer.run()


def gameplay_worker(ports, args):
    """Wrapper around a simple gameplay worker.

    Args:
        port (int): Port server is listening on.
        args (dict): args dict.
    """
    model = InferenceClient(ports)
    data_client = DataGameClient(args)
    game_gen = GameGenerator(
        args.board_size,
        model,
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
        servers = []
        ports = list(range(args.inference_server_port, args.num_servers))
        for port in ports:
            servers.append(
                Process(target=inference_worker, args=(port, args), daemon=True)
            )

        for server in servers:
            server.start()

        # Start the databroker
        broker = Process(target=databroker_worker, args=(args,), daemon=True)
        broker.start()

        # Start the trainer
        trainer = Process(target=trainer_worker, args=(args,))
        trainer.start()

        # start the games
        games = []
        for _ in range(args.num_games):
            games.append(
                Process(target=gameplay_worker, args=(ports, args), daemon=True)
            )
        for g in games:
            g.start()

        for s in servers:
            s.join()
        broker.join()
        trainer.join()
        for g in games:
            g.join()

    except KeyboardInterrupt:
        pass
    finally:
        print("Exiting...")
