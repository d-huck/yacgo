""""""

import signal
import time
from multiprocessing import Process

import wandb
from yacgo.data import DataBroker
from yacgo.utils import make_args


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


def main():
    """
    Main Process
    """
    args = make_args()
    db = Process(target=databroker_worker, args=(args,))
    db.start()
    try:
        db.join()
    except KeyboardInterrupt:
        db.terminate()
        db.join()


if __name__ == "__main__":
    main()
