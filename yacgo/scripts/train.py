"""
Example script on running a simple training server
"""

from multiprocessing import Process
from yacgo.utils import make_args
from yacgo.models import Trainer
from yacgo.data import DataBroker


def trainer_worker(args):
    """Wrapper around a simple trainer worker.

    Args:
        args (dict): args dict.
    """
    trainer = Trainer(args)
    print("Starting trainer...")
    trainer.run()


def databroker_worker(args):
    """Wrapper around a simple databroker worker.

    Args:
        port (int): Port server is listening on.
        args (dict): args dict.
    """
    broker = DataBroker(args)
    print("Starting databroker...")
    broker.run()


def main():
    """
    Main Process
    """
    args = make_args()
    try:
        trainer = Process(target=trainer_worker, args=(args), daemon=True)
        trainer.start()
        databroker = Process(target=databroker_worker, args=(args), daemon=True)
        databroker.start()

        trainer.join()
        databroker.join()

    except KeyboardInterrupt:
        pass
    finally:
        print("Exiting...")


if __name__ == "__main__":
    main()
