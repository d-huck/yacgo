"""
Utility functions
"""

import argparse
import signal
from typing import Tuple

import msgpack
import numpy as np
import torch

from yacgo.data import TORCH_DTYPE, DATA_DTYPE  # pylint: disable=unused-import


def build_logger():
    pass


def init_signals():
    """Ignore SIGINT in child workers."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def pack_state(states: np.ndarray) -> bytearray:
    """Packs a state for transmission across zmq sockets

    Args:
        states (np.ndarray): state of shape (bs, n_channels, board_size, board_size)

    Returns:
        bytearray: packed message
    """
    return msgpack.packb((states.shape, states.tobytes()))


def unpack_state(message: bytearray) -> np.ndarray:
    """
    Unpacks state for inference

    Args:
        message (bytearray): incoming byte array with state information

    Returns:
        np.ndarray: state with shape (bs, n_channels, board_size, board_size)
    """
    shape, state = msgpack.unpackb(message)
    return np.frombuffer(state, DATA_DTYPE).reshape(shape)


# TODO: make Inference and State data classes
def pack_inference(values: np.ndarray, policies: np.ndarray) -> bytearray:
    """
    Packs inference results for transmission across zmq sockets

    Args:
        values (np.ndarray): values of shape (bs, )
        policies (np.ndarray): policies of shape (bs, board_size ** 2 + 1)

    Returns:
        bytearray: packed message
    """
    v = (values.shape, values.tobytes())
    p = (policies.shape, policies.tobytes())

    return msgpack.packb((v, p))


def unpack_inference(message: bytearray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unpacks inference result for use by MCTS tree search

    Args:
        message (bytearray): packed message

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of values and policies with respective
        shapes (bs, ) and (bs, board_size ** 2 + 1)
    """
    v, p = msgpack.unpackb(message)
    values = np.frombuffer(v[1], DATA_DTYPE).reshape(v[0])[0]
    policies = np.frombuffer(p[1], DATA_DTYPE).reshape(p[0])
    return values, policies


def pack_examples(
    states: np.ndarray, values: np.ndarray, policies: np.ndarray
) -> bytearray:
    """
    Packs a training example for transmission to data broker

    Args:
        states (np.ndarray): state of shape (bs, n_channels, board_size, board_size)
        values (np.ndarray): values of shape (bs, )
        policies (np.ndarray): policies of shape (bs, board_size ** 2 + 1)

    Returns:
        bytearray: packed message
    """
    s = (states.shape, states.tobytes())
    v = (values.shape, values.tobytes())
    p = (policies.shape, policies.tobytes())

    return msgpack.packb((s, v, p))


def unpack_examples(message: bytearray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unpacks message for storage in databroker

    Args:
        message (bytearray): packed message

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of (states, values,
        policies) with respective shapes (bs, n_channels, board_size, board_size),
        (bs, ), (bs, board_size ** 2 + 1)
    """
    s, v, p = msgpack.unpackb(message)
    states = np.frombuffer(s[1], DATA_DTYPE).reshape(s[0])
    values = np.frombuffer(v[1], DATA_DTYPE)
    policies = np.frombuffer(p[1], DATA_DTYPE).reshape(p[0])

    return (states, values, policies)


def make_args():
    """Function to hold arguments for all scripts creating a unified
    argument structure

    Returns:
        dict: args
    """
    parser = argparse.ArgumentParser()
    device = "cuda" if torch.cuda.is_available() else "mps"

    # training args
    parser.add_argument(
        "--device",
        type=str,
        default=str(device),
        help="Device for inference/training. Defaults to CUDA, MPS recommended on MacOS",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=0,
        help="Starting epoch for training. Defaults to 0 to start training from scratch",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="S0",
        help="Size of the model. Options are S0, S1, S2, L",
    )
    parser.add_argument(
        "--training_batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--training_steps_per_epoch",
        type=int,
        default=10_000,
        help=(
            "How many training steps per epoch. At the end of each epoch,"
            "a competition is run to update the model."
        ),
    )
    parser.add_argument(
        "--game_records",
        type=str,
        default="game_records.csv",
        help="File to store game records. Fails silently if does not exist",
    )
    parser.add_argument(
        "--competition_epochs",
        type=int,
        default=5,
        help="Number of epochs between competitions. Defaults to 5",
    )
    parser.add_argument(
        "--num_feature_channels",
        "-fc",
        type=int,
        default=12,
        help="Number of feature channels for the model. Defaults to 12.",
    )
    parser.add_argument(
        "--weights_cache_dir",
        type=str,
        default=None,
        help="Directory to cache model weights. Defaults to None",
    )
    parser.add_argument(
        "--refill_buffer",
        type=bool,
        default=True,
        help="Whether to refill the replay buffer after training on a state. Defaults to False",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for training"
    )

    # Inference settings
    parser.add_argument(
        "--inference_batch_size",
        type=int,
        default=128,
        help=(
            "Max batch size for inference. Due to multiple servers being used,"
            "actual batch size will usually be lower to minimize time a game is "
            "waiting for a server to finish. Defaults to 128.",
        ),
    )
    parser.add_argument(
        "--inference_server_port",
        type=int,
        default=5000,
        help=(
            "Starting port for the inference servers. Inference servers will run on"
            "ports 5000 through 5000 + N."
        ),
    )
    parser.add_argument(
        "--inference_server_address",
        type=str,
        default="localhost",
        help=("Address of the inference servers.",),
    )
    parser.add_argument(
        "--num_servers", type=int, default=2, help="Number of inference servers to use"
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to model weights"
    )

    # Data and weight sharing settings
    parser.add_argument(
        "--databroker_port",
        type=int,
        default=6000,
        help="Port for the databroker server. Defaults to 6000",
    )
    parser.add_argument(
        "--databroker_address",
        type=str,
        default=None,
        help=(
            "Address for the databroker server. Defaults to None, which"
            "listens on all public addresses.",
        ),
    )
    parser.add_argument(
        "--replay_buffer_size",
        type=int,
        default=100_000,
        help="Size of the replay buffer. Defaults to 100,000",
    )
    parser.add_argument(
        "--replay_buffer_min_size",
        type=int,
        default=10_000,
        help="Minimum size of the replay buffer before training starts",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models/",
        help="Directory to store models that beat the previous best.",
    )
    parser.add_argument(
        "--data_cache_dir",
        type=str,
        default=".data_cache/",
        help="Directory to cache game data. Defaults to None which does not cache",
    )
    parser.add_argument(
        "--model_server_port",
        type=int,
        default=6010,
        help="Port for the model server. Defaults to 6010",
    )
    parser.add_argument(
        "--model_server_address",
        type=str,
        default=None,
        help=(
            "Address for the model server. Defaults to None, which"
            "listens on all public addresses.",
        ),
    )
    parser.add_argument(
        "--forget_rate",
        type=float,
        default=0.15,
        help="Rate at which to forget old data. Defaults to 0.15",
    )

    # Game / MCTS Settings
    parser.add_argument(
        "--num_games", type=int, default=8, help="Number of games to play"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--num_game_processes",
        type=int,
        default=16,
        help=(
            "Number of game processes. Each process will run num_games // num_processes threads "
        ),
    )
    parser.add_argument(
        "--num_comp_games",
        type=int,
        default=32,
        help="Number of games to play for competition",
    )
    parser.add_argument(
        "--komi",
        type=float,
        default=0.5,
        help="Komi for Go game. Defaults to 0.5 to avoid draws, giving white slight advantage",
    )
    parser.add_argument(
        "--starting_elo",
        type=float,
        default=0.0,
        help="Starting elo for model. Defaults to 0.0",
    )
    parser.add_argument(
        "--n_simulations",
        type=int,
        default=1200,
        help="Number of simulations for MCTS for gameplay. Defaults to 1200",
    )
    parser.add_argument(
        "--pcap_train",
        type=int,
        default=None,
        help=(
            "Number of simulations for MCTS during data generation. defaults "
            "to n_simulations (600) if not provided."
        ),
    )
    parser.add_argument(
        "--pcap_fast",
        type=int,
        default=100,
        help="Number of of fast for MCTS. Defaults to 100",
    )
    parser.add_argument(
        "--pcap_prob",
        type=float,
        default=0.25,
        help="Probability of using Playout Cap in training. Defaults to 0.25",
    )

    parser.add_argument(
        "--board_size",
        type=int,
        default=19,
        help="Board size for Go game. Defaults to 19x19",
    )
    parser.add_argument(
        "--c_puct",
        type=float,
        default=1.1,
        help="C_puct for MCTS. Defaults to 1.1",
    )
    parser.add_argument(
        "--mcts_noise",
        type=bool,
        default=True,
        help="Whether to sample from Direchlet noise in MCTS when generating games. Defaults to False",
    )
    parser.add_argument(
        "--softmax_temp",
        type=float,
        default=1.2,
        help="softmax temp applied to policy at root during training",
    )
    parser.add_argument(
        "--train_random_symmetry",
        type=bool,
        default=True,
        help="Whether to apply random symmetries to states during training",
    )
    parser.add_argument(
        "--acceptance_ratio",
        type=float,
        default=0.55,
        help="Percentage of games a model needs to win to become new best model. Defaults to 0.55",
    )
    parser.add_argument(
        "--max_priority",
        type=float,
        default=1_000_000,
        help="Maximum priority for replay buffer. Defaults to 1_000_000",
    )

    # Misc
    parser.add_argument(
        "--wandb",
        type=bool,
        default=True,
        help="Whether to log training to wandb. Defaults to True",
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="yacgo",
        help="Wandb project name. Defaults to yacgo",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="Wandb entity name. Defaults to None which will create a random group name",
    )
    parser.add_argument(
        "--global_step",
        type=int,
        default=0,
        help="Global step, useful for resuming training and sane logging. Defaults to 0",
    )

    args = parser.parse_args()

    if args.num_games < args.num_game_processes:
        args.num_game_processes = args.num_games

    # Do any sanitation here
    if args.data_cache_dir is not None and not args.data_cache_dir.endswith("/"):
        args.data_cache_dir += "/"
    if args.weights_cache_dir is not None and not args.weights_cache_dir.endswith("/"):
        args.weights_cache_dir += "/"
    if args.pcap_train is None:
        args.pcap_train = args.n_simulations

    return args


def set_args(**kwargs):
    args = make_args()
    d = vars(args)
    for k, v in kwargs.items():
        d[k] = v

    return args


def model_name_to_epoch(model: str) -> int:
    """Converts a string number with a zero prefix to an integer.
    Assumes the model is named in the format "xxxxx-xxx-xx-0000.pth"

    Args:
        num (str): string number with a zero prefix

    Returns:
        int: integer representation of the string number
    """
    epoch_str = model.split("-")[-1][:-4].lstrip("0")
    if epoch_str == "":
        return 0
    return int(epoch_str)
