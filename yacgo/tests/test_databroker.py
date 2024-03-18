"""
Simple script to test databroker flow
"""

from multiprocessing import Process
from yacgo.models import Trainer
from yacgo.data import DataBroker, KataGoDataClient
from yacgo.utils import make_args


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


def main():
    """
    Main Process
    """
    args = make_args()
    try:
        databroker = Process(target=databroker_worker, args=(args,), daemon=True)
        databroker.start()
        trainer = Process(target=trainer_worker, args=(args,), daemon=True)
        trainer.start()
        data = KataGoDataClient(args)
        data.run("data/kata1-b40c256x2-s5095420928-d1229425124")

        trainer.join()
        databroker.join()

    except KeyboardInterrupt:
        pass
    finally:
        print("Exiting...")


if __name__ == "__main__":
    main()
