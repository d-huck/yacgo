"""
Utility functions
"""

import argparse


def make_args():
    """Function to hold arguments for all scripts creating a unified
    argument structure

    Returns:
        dict: args
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference/training. Defaults to CUDA, MPS recommended on MacOS",
    )
    parser.add_argument(
        "--board_size",
        type=int,
        default=19,
        help="Board size for Go game. Defaults to 19x19",
    )
    parser.add_argument(
        "--inference_batch_size",
        type=int,
        default=128,
        help="Max batch size for inference. Due to multiple servers being used, actual batch size will usually be lower to minimize time a game is waiting for a server to finish. Defaults to 128.",
    )
    parser.add_argument(
        "--num_servers", type=int, default=2, help="Number of inferenceservers to use"
    )
    parser.add_argument(
        "--num_games", type=int, default=128, help="Number of games to play"
    )
    parser.add_argument(
        "--num_feature_channels",
        "-fc",
        type=int,
        default=12,
        help="Number of feature channels for the model",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="S0",
        help="Size of the model. Options are S0, S1, S2, L",
    )

    args = parser.parse_args()
    return args
