"""
Runs the entire yacgo stack: 1 Trainer, 1 DataBroker, n InferenceServers, k Gameplay clients.
"""

from multiprocessing import Process
from yacgo.utils import make_args
from yacgo.models import InferenceServer, Trainer
from yacgo.databroker import DataBroker


def inference_worker(port, args):
    """Wrapper around a simple inference worker.

    Args:
        port (int): Port server is listening on.
        args (dict): args dict.
    """
    server = InferenceServer(port, args)
    print("Starting server...")
    server.run()


def databroker_worker(port, args):
    """Wrapper around a simple databroker worker.

    Args:
        port (int): Port server is listening on.
        args (dict): args dict.
    """
    broker = DataBroker(port, args)
    print("Starting databroker...")
    broker.run()


def trainer_worker(port, args):
    """Wrapper around a simple trainer worker.

    Args:
        args (dict): args dict.
    """
    trainer = Trainer(port, args)
    print("Starting trainer...")
    trainer.run()


def main():
    """
    Main Process
    """
    args = make_args()
    try:
        servers = []
        port = args.inference_server_port
        for _ in range(args.num_servers):
            servers.append(
                Process(target=inference_worker, args=(port, args), daemon=True)
            )
            port += 1

        for server in servers:
            server.start()

        # Start the databroker
        broker = Process(
            target=databroker_worker, args=(args.databroker_port, args), daemon=True
        )

        # Start the trainer
        trainer = Process(target=trainer_worker, args=(args,))
    except KeyboardInterrupt:
        pass
    finally:
        print("Exiting...")
