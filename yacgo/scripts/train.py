"""
Example script on running a simple training server
"""

import multiprocessing as mp
import time
from multiprocessing import Process

from yacgo.utils import make_args
from yacgo.models import Trainer, InferenceClient
from yacgo.data import DataBroker
from yacgo.train_utils import CompetitionResult, ModelCompetition


def trainer_worker(args):
    """Wrapper around a simple trainer worker.

    Args:
        args (dict): args dict.
    """
    trainer = Trainer(args)
    print("Starting trainer...")
    count = 1
    while True:
        trainer.run()
        trainer.save_pretrained(f"/data/models/yacgo/5x5-initial-epoch{count:03d}")
        count += 1
        # ports = list(
        #     range(
        #         args.inference_server_port,
        #         args.inference_server_port + args.num_servers,
        #     )
        # )
        # inf = InferenceClient(ports)
        # time.sleep(5)
        # inf.model = trainer.model
        # comp = ModelCompetition(args.board_size, inf, None, args)
        # result = comp.compete(num_games=128)
        # print(result)
        # break


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
    mp.set_start_method("forkserver")
    try:
        trainer = Process(target=trainer_worker, args=(args,), daemon=True)
        trainer.start()
        databroker = Process(target=databroker_worker, args=(args,), daemon=True)
        databroker.start()

        trainer.join()
        databroker.join()

    except KeyboardInterrupt:
        pass
    finally:
        print("Exiting...")


if __name__ == "__main__":
    main()
