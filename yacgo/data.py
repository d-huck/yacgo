"""
Implements a databroker class which handles the storing of data, construction of 
a replay buffer, and serving data to the trainer.

All inputs to the DataBroker should consist of a message type and a message. The 
message type is used to determine the action to take with the message. 

- In the case where the message type is DEPOSIT, the message is stored in the 
replay buffer. The expected message should be a tuple of (state, value, policy). 
Each item in the tuple should have the same _first_  dimension.

- In the case where the message type is TRAINING_BATCH, the message is used to 
request a batch of data from the replay buffer. The expected message should be 
the same as above.
"""

import logging
import os
import uuid
from dataclasses import dataclass, field
from queue import Empty, PriorityQueue
from random import randint
from typing import Union

import msgpack
import numpy as np
import torch
import zmq

DEPOSIT = 0
GET_BATCH = 0
RESET = 1

HIGH_PRIORITY = 50
TRAINING_BATCH = 1
QUIT = -1

DATA_DTYPE = np.float32  # pylint: disable=C0103
TORCH_DTYPE = torch.float32

logger = logging.getLogger(__name__)


@dataclass  # Still use data class to get the extra __repr__ and __eq__ methods for free
class TrainState:
    """Dataclass for a single training example. Contains a state, value, and policy."""

    def __init__(self, state: np.ndarray, value: np.float32, policy: np.ndarray):
        if isinstance(value, (int, float)):
            value = DATA_DTYPE(value)
        elif isinstance(value, list):
            value = np.array(value, dtype=DATA_DTYPE)
        if len(value.shape) == 0:
            value = value
        elif len(value.shape) == 1 and value.shape[0] == 1:
            value = value[0]
        else:
            raise ValueError("Value must be a single number")
        self.state = state
        self.value = value
        self.policy = policy

    def pack(self) -> bytearray:
        """Packs the TrainState into a zmq message.

        Returns:
            bytearray: message for zmq
        """
        s = (self.state.shape, self.state.tobytes())
        v = (self.value.shape, self.value.tobytes())
        p = (self.policy.shape, self.policy.tobytes())

        return msgpack.packb((s, v, p))

    @classmethod
    def unpack(cls, message: bytearray) -> "TrainState":
        """Create a TrainState from a zmq message.

        Args:
            message (bytearray): message from zmq

        Returns:
            TrainState: with information from the message
        """
        s, v, p = msgpack.unpackb(message)

        state = np.frombuffer(s[1], DATA_DTYPE).reshape(s[0])
        value = np.frombuffer(v[1], DATA_DTYPE)
        policy = np.frombuffer(p[1], DATA_DTYPE).reshape(p[0])
        return cls(state, value, policy)


@dataclass
class TrainingBatch:
    """Dataclass for a training batch, identical to TrainState, except takes an
    array for values and keeps track of batch size."""

    batch_size: int
    states: np.ndarray
    values: np.ndarray
    policies: np.ndarray

    def pack(self) -> bytearray:
        """
        Packs the TrainingBatch into a zmq message.

        Returns:
            bytearray: message for zmq
        """
        s = (self.states.shape, self.states.tobytes())
        v = (self.values.shape, self.values.tobytes())
        p = (self.policies.shape, self.policies.tobytes())

        return msgpack.packb((s, v, p))

    @classmethod
    def unpack(cls, message) -> "TrainingBatch":
        """
        Create a TrainingBatch from a zmq message.

        Args:
            message (bytearray): message from zmq

        Returns:
            TrainingBatch: with information from the message
        """
        s, v, p = msgpack.unpackb(message)
        states = np.frombuffer(s[1], DATA_DTYPE).reshape(s[0])
        values = np.frombuffer(v[1], DATA_DTYPE).reshape(v[0])
        policies = np.frombuffer(p[1], DATA_DTYPE).reshape(p[0])
        return cls(states.shape[0], states, values, policies)


@dataclass
class State:
    """Simple data class to hold state from gameplay"""

    state: np.ndarray

    def pack(self) -> bytearray:
        """Packs the state into zmq message

        Returns:
            bytearray: message for zmq
        """
        s = (self.state.shape, self.state.tobytes())
        return msgpack.packb(s)

    @classmethod
    def unpack(cls, message: bytearray) -> "State":
        """Unpack the state into State object

        Args:
            message (bytearray): incoming message from zmq

        Returns:
            State
        """
        s = msgpack.unpackb(message)
        state = np.frombuffer(s[1], DATA_DTYPE).reshape(s[0])
        return cls(state)


@dataclass
class Inference:
    """Simple data class to hold inferences from inference server"""

    def __init__(self, value: Union[DATA_DTYPE, np.ndarray], policy: np.ndarray):
        if len(value.shape) == 0:
            value = value.astype(DATA_DTYPE)
        elif len(value.shape) == 1 and value.shape[0] == 1:
            value = value[0].astype(DATA_DTYPE)
        else:
            raise ValueError("Value must be a single number")
        self.value = value.copy()
        self.policy = policy.copy()

    def pack(self) -> bytearray:
        """Packs the inference into a zmq message

        Returns:
            bytearray: message for zmq
        """
        v = (self.value.shape, self.value.tobytes())
        p = (self.policy.shape, self.policy.tobytes())

        return msgpack.packb((v, p))

    @classmethod
    def unpack(cls, message: bytearray) -> "Inference":
        """Unpacks a message from zmq and constructs an Inference class containing
        the information

        Args:
            message (bytearray): incoming message

        Returns:
            Inference
        """
        v, p = msgpack.unpackb(message)
        if len(v[0]) == 0:
            value = np.frombuffer(v[1], DATA_DTYPE)
        else:
            value = np.frombuffer(v[1], DATA_DTYPE).reshape(v[0])[
                0
            ]  # ensure value is single number
        policy = np.frombuffer(p[1], DATA_DTYPE).reshape(p[0])

        return cls(value, policy)


@dataclass(order=True)
class PrioritizedTrainState(object):
    """
    Because items are placed into the replay buffer and trained on only once,
    this class creates a pseudo-shuffling by giving each training example a
    randomized priority. This will ensure that samples from the same game have a low
    probability of being clumped together in training data without having to shuffle
    the replay buffer every time a game is added to it.
    """

    priority: int
    state: np.ndarray = field(compare=False)
    value: DATA_DTYPE = field(compare=False)
    policy: np.ndarray = field(compare=False)

    @classmethod
    def from_train_state(cls, s: TrainState) -> "PrioritizedTrainState":
        "Create a prioritized data from a TrainState"
        return cls(
            randint(0, HIGH_PRIORITY), s.state.copy(), s.value.copy(), s.policy.copy()
        )


class DataBroker(object):
    """
    Databroker class to handle the consolidation and serving of data to the
    Trainer.
    """

    def __init__(
        self,
        args,
    ):
        self.min_size = args.replay_buffer_min_size
        self.max_size = args.replay_buffer_size
        self.port = args.databroker_port
        self.replay_buffer = PriorityQueue()
        self.cache_dir = args.data_cache_dir
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(f"tcp://*:{self.port}")
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, 1)
        self.refill_buffer = args.refill_buffer
        self.batch_size = args.training_batch_size
        self.min_size = (
            args.training_batch_size
            if not self.refill_buffer
            else 16 * args.training_batch_size
        )
        self.max_priority = (
            self.batch_size * args.training_steps_per_epoch * HIGH_PRIORITY * 2
        )
        self.train_random_symmetry = args.train_random_symmetry
        self.forget_rate = args.forget_rate

    @staticmethod
    def random_symmetry(state, value, policy):
        pol_pass = policy[-1]
        policy_1d = policy[:-1]
        policy_2d = np.reshape(policy_1d, (state.shape[1], state.shape[1]))

        orientation = np.random.randint(0, 8)
        if (orientation >> 0) % 2:
            # Horizontal flip
            state = np.flip(state, 2)
            policy_2d = np.flip(policy_2d, 1)
        if (orientation >> 1) % 2:
            # Vertical flip
            state = np.flip(state, 1)
            policy_2d = np.flip(policy_2d, 0)
        if (orientation >> 2) % 2:
            # Rotate 90 degrees
            state = np.rot90(state, axes=(1, 2))
            policy_2d = np.rot90(policy_2d, axes=(0, 1))

        policy_1d = np.ndarray.flatten(policy_2d)
        policy = np.append(policy_1d, pol_pass)
        return state, value, policy

    def get_batch(self) -> TrainingBatch:
        """Returns a single batch from the replay buffer

        Args:
            batch_size (int, optional): Batch size. Defaults to 32.

        Returns:
            TrainingBatch: Training Batch
        """
        if self.replay_buffer.qsize() < self.min_size:
            return TrainingBatch(0, np.array([]), np.array([]), np.array([]))

        states = []
        values = []
        policies = []
        count = 0
        for _ in range(self.batch_size):
            try:
                data = self.replay_buffer.get(block=False)
            except Empty:  # python can be so gross sometimes
                break

            if self.train_random_symmetry:
                state, value, policy = DataBroker.random_symmetry(
                    data.state, data.value, data.policy
                )
            else:
                state, value, policy = data.state, data.value, data.policy

            states.append(state)
            values.append(value)
            policies.append(policy)
            if self.refill_buffer:
                data.priority += HIGH_PRIORITY  # put at the end of the queue
            states.append(data.state)
            values.append(data.value)
            policies.append(data.policy)
            if self.refill_buffer and np.random.rand() > self.forget_rate:
                # put at the end of the queue and add some randomization of order
                data.priority += (
                    HIGH_PRIORITY + randint(-HIGH_PRIORITY, HIGH_PRIORITY) // 4
                )
                self.replay_buffer.put(data)
            count += 1

        states = np.stack(states)
        values = np.stack(values)
        policies = np.stack(policies)

        batch = TrainingBatch(count, states, values, policies)
        return batch

    def load_from_disk(self):
        """
        Loads cached data from disk

        Args:
            path (_type_): _description_
        """
        if self.cache_dir is None:
            return

        for file in os.listdir(self.cache_dir):
            if file.endswith(".npz"):
                data = np.load(file)
                states = data["states"]
                values = data["values"]
                policies = data["policies"]
            for s, v, p in zip(states, values, policies):
                self.replay_buffer.put(
                    PrioritizedTrainState(randint(0, HIGH_PRIORITY), s, v, p)
                )

    def dump_to_disk(self):
        """
        Dumps data to disk for caching

        Args:
            batch (_type_): _description_
            path (_type_): _description_
        """
        if self.cache_dir is None:
            return

        self.refill_buffer = False
        while not self.replay_buffer.empty():
            batch = self.get_batch()
            fp = self.cache_dir + str(uuid.uuid4()) + ".npz"
            np.savez_compressed(
                fp, states=batch.states, values=batch.values, policies=batch.policies
            )

    def reset(self, save=False):
        if save:
            self.dump_to_disk()
        self.replay_buffer = PriorityQueue()

    def process_data(self, message):
        """
        Unpacks data from a message and places it in the replay buffer

        Args:
            message (_type_): _description_

        Returns:
            _type_: _description_
        """
        example = TrainState.unpack(message)
        data = PrioritizedTrainState.from_train_state(example)
        self.replay_buffer.put(data)

    def run(self):
        """
        Starts the data broker
        """
        while True:
            try:
                address, _, message = self.socket.recv_multipart()

                if str(address, "utf-8").startswith("TRAIN"):
                    msg = int(msgpack.unpackb(message))
                    if msg == GET_BATCH:
                        batch = self.get_batch()
                        if batch is not None:
                            self.socket.send_multipart([address, b"", batch.pack()])
                        else:
                            self.socket.send_multipart([address, b"", b""])
                    elif msg == RESET:
                        self.reset(save=True)
                        self.socket.send_multipart([address, b"", b""])
                    else:
                        self.socket.send_multipart([address, b"", b""])
                else:
                    self.process_data(message)
                    self.socket.send_multipart([address, b"", b""])
            except zmq.error.Again:
                pass
            except KeyboardInterrupt:
                break
            print(
                "Data Buffer size: ",
                self.replay_buffer.qsize(),
                end="      \r",
            )
        print(
            f"Databroker has {self.replay_buffer.qsize()} items in the queue, dumping to disk..."
        )
        self.dump_to_disk()
        self.socket.close()
        self.context.term()
        print("DataBroker closed")


class DataGameClientMixin:
    """Client for a GameGenerator to interact with the Databroker"""

    def __init__(self, args: dict):
        identity = f"GAME-{uuid.uuid4()}".encode()
        self.data_context = zmq.Context.instance()
        self.data_socket = self.data_context.socket(zmq.REQ)
        self.data_socket.setsockopt(zmq.IDENTITY, identity)
        self.data_socket.connect(
            f"tcp://{args.inference_server_address}:{args.databroker_port}"
        )

    def deposit(self, example: TrainState):
        """Deposits a single training example into the data broker"""
        self.data_socket.send(example.pack())
        _ = self.data_socket.recv()

    def destroy(self):
        """Sanely shuts down the zmq client"""
        self.data_socket.close()
        self.data_context.term()
        print("DataTrainClient closed")


class DataTrainClientMixin:
    """Client for a Trainer to interact with the Databroker"""

    def __init__(self, args: dict):
        self.batch_size = args.training_batch_size
        identity = f"TRAIN-{uuid.uuid4()}".encode()
        self.data_context = zmq.Context.instance()
        self.data_socket = self.data_context.socket(zmq.REQ)
        self.data_socket.setsockopt(zmq.IDENTITY, identity)
        self.data_socket.connect(f"tcp://localhost:{args.databroker_port}")

    def get_batch(
        self,
    ) -> Union[TrainingBatch, None]:
        """Returns a TrainingBatch from the data broker"""
        self.data_socket.send(msgpack.packb(GET_BATCH))
        return TrainingBatch.unpack(self.data_socket.recv())

    def reset_data(self):
        """Resets the data broker"""
        self.data_socket.send(msgpack.packb(RESET))
        _ = self.data_socket.recv()

    def destroy(self):
        """Sanely shuts down the zmq client"""
        self.data_socket.close()
        self.data_context.term()
        print("DataTrainClient closed")


class KataGoDataClient(DataGameClientMixin):
    """
    DataClient that reads KataGo game files from disks and deposits them to the
    DataBroker
    """

    @staticmethod
    def read_game(file_path: str):
        """
        Reads a single KataGo game file and returns the data

        Args:
            file_path (str): Path to the KataGo game file

        Returns:
            Tuple[np.ndarray, np.float32, np.ndarray]: state, value, policy
        """
        data = np.load(file_path)
        policies = data["policyTargetsNCMove"].astype(DATA_DTYPE)
        board_size = int(np.sqrt(policies.shape[-1] - 1))
        states = data["binaryInputNCHWPacked"]
        states = np.unpackbits(states, axis=2)
        states = states[:, :, : board_size * board_size]
        states = states.reshape(
            states.shape[0], states.shape[1], board_size, board_size
        ).astype(DATA_DTYPE)
        values = data["globalTargetsNC"].astype(DATA_DTYPE)

        for s, v, p in zip(states, values, policies):
            value = (v[0] * 2 - 1).astype(DATA_DTYPE)
            out = TrainState(s, value, p[0])
            yield out

    def run(self, file_path: str):
        """
        Reads a directory of KataGo game files and deposits the data into the
        DataBroker

        Args:
            file_path (str): Path to the KataGo game file
        """
        for file in os.listdir(file_path):
            if file.endswith(".npz"):
                fp = os.path.join(file_path, file)
                for example in self.read_game(fp):
                    self.deposit(example)
