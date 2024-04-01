import torch
import torch.nn as nn

from yacgo.models import Trainer
from yacgo.utils import make_args


def main():
    args = make_args()
    trainer = Trainer(args)

    example = torch.randn(
        1024, args.num_feature_channels, args.board_size, args.board_size
    ).to(args.device)
    try:
        trainer.model(example)
    except RuntimeError as e:
        print(e)
        print("Model not implemented yet.")


if __name__ == "__main__":
    main()
