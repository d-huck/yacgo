"""
Example script on running a simple training server
"""

import multiprocessing as mp
import time
from multiprocessing import Process


from yacgo.utils import make_args
from yacgo.models import Trainer, InferenceClient, InferenceLocal
from yacgo.data import DataBroker
from yacgo.train_utils import CompetitionResult, ModelCompetition


def competition(args):
    model1 = InferenceLocal(args, "/data/models/yacgo/5x5-initial-epoch101")
    model2 = InferenceLocal(args, "/data/models/yacgo/5x5-initial-epoch055")
    print("Loading inference local worker")
    comp = ModelCompetition(args.board_size, model1, model2, args)
    result = comp.compete(num_games=16)
    del model1, model2
    print(result)


def trainer_worker(args):
    """Wrapper around a simple trainer worker.

    Args:
        args (dict): args dict.
    """
    trainer = Trainer(args)

    count = 1
    while True:
        trainer.run()
        trainer.save_pretrained(
            f"/data/models/yacgo/ver2/initial_trials-bs{args.board_size}-e{count:03d}.tgz"
        )
        count += 1


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
