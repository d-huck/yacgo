"""
Utility functions
"""

import argparse
import signal
from typing import Tuple

import msgpack
import numpy as np
import torch

DATA_DTYPE = np.float32  # pylint: disable=C0103
TORCH_DTYPE = torch.float32


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

    parser.add_argument(
        "--device",
        type=str,
        default=str(device),
        help="Device for inference/training. Defaults to CUDA, MPS recommended on MacOS",
    )
    parser.add_argument(
        "--board_size",
        type=int,
        default=19,
        help="Board size for Go game. Defaults to 19x19",
    )
    parser.add_argument(
        "--training_batch_size", type=int, default=128, help="Batch size for training"
    )
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
        "--num_servers", type=int, default=2, help="Number of inference servers to use"
    )

    parser.add_argument(
        "--num_games", type=int, default=4, help="Number of games to play"
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
        "--databroker_port",
        type=int,
        default=6000,
        help="Port for the databroker server. Defaults to 6000",
    )

    parser.add_argument(
        "--num_feature_channels",
        "-fc",
        type=int,
        default=6,
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
