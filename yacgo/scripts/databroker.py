from yacgo.data import DataBroker
from yacgo.utils import make_args


def main():
    """
    Main Process
    """
    args = make_args()
    databroker = DataBroker(args)
    databroker.run()


if __name__ == "__main__":
    main()
