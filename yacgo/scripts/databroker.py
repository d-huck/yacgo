import atexit
import time
from yacgo.data import DataBroker
from yacgo.utils import make_args


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
    databroker = DataBroker(args)
    databroker.run()


if __name__ == "__main__":
    main()
