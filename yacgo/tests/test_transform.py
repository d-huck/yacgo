"""Tests the transform in TrainState directly and implicitly checks InferenceClient"""

import numpy as np
import torch
from torchvision import transforms

from yacgo.data import TrainState
from yacgo.go import game, govars

transform = transforms.Compose(
    [
        transforms.RandomCrop(19, padding=0, pad_if_needed=True),
    ]
)


def test_TrainState():
    states = [
        game.init_state(3),
        game.init_state(5),
        game.init_state(9),
        game.init_state(13),
        game.init_state(19),
    ]

    for state in states:
        value = np.random.random()
        policy = np.random.random(game.action_size(state))
        s = TrainState(state, value, policy)
        s.transform()
        assert s.state.shape == (govars.NUM_CHNLS, 19, 19)
        assert s.policy.shape == (game.action_size(s.state),)
        board_mask = np.append(s.state[govars.BOARD_MASK].ravel(), 1)
        assert board_mask.shape == (19 * 19 + 1,)
        _policy = np.zeros_like(s.policy)
        _policy = s.policy[board_mask == 1]
        assert np.array_equal(_policy, policy), "Policy not preserved."
        s.pack()


def main():
    test_TrainState()
    print("Success!")


if __name__ == "__main__":
    main()
