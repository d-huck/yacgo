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
    # try:
    trainer.model.eval()
    print("testing eval")
    trainer.model.forward(example)
    trainer.model.train()
    trainer.model.forward(example)
    print("testing train")
    # except RuntimeError as e:
    #     print(e)
    #     print("Model not implemented yet.")

    print("Success!")


if __name__ == "__main__":
    main()
