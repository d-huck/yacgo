import signal
from multiprocessing import Process

from yacgo.models import InferenceServer
from yacgo.eval import ModelCompetition
from yacgo.utils import make_args

CANDIDATE_PORT = 31338


def inference_worker(port, args):
    """Wrapper around a simple inference worker.

    Args:
        port (int): Port server is listening on.
        args (dict): args dict.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    server = InferenceServer(port, args)
    server.run()


def main():
    args = make_args()
    server = Process(target=inference_worker, args=(CANDIDATE_PORT, args), daemon=True)
    server.start()

    comp = ModelCompetition(CANDIDATE_PORT, None, args)
    result = comp.compete(100)

    print(result)
    server.terminate()
    server.join()


if __name__ == "__main__":
    main()
