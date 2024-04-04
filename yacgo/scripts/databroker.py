""""""

import atexit
import time
from yacgo.data import DataBroker
from yacgo.utils import make_args
import wandb


@atexit.register
def cleanup_wait():
    """
    Cleanup function
    """
    time.sleep(5)
    print("Waiting for cleanup...")


def main():
    """
    Main Process
    """
    args = make_args()
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            job_type="replay_buffer",
            config=args,
        )
    databroker = DataBroker(args)
    try:
        databroker.run()
    except KeyboardInterrupt:
        print("Dumping to disk and exiting...")


if __name__ == "__main__":
    main()
