"""
Runs a yacgo inference server.
"""

from multiprocessing import Process

from yacgo.models import InferenceServer
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


def main():
    """Main process"""
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

        for server in servers:
            server.join()
    except KeyboardInterrupt:
        pass
    finally:
        print("Exiting...")


if __name__ == "__main__":
    main()
