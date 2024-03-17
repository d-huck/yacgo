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

import os
import uuid
from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Union
from random import randint

import msgpack
import numpy as np
import zmq

from yacgo.train_utils import TrainState
from yacgo.utils import DATA_DTYPE

DEPOSIT = 0
HIGH_PRIORITY = 50
TRAINING_BATCH = 1
QUIT = -1


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
        states = np.frombuffer(s[1], np.float32).reshape(s[0])
        values = np.frombuffer(v[1], np.float32).reshape(v[0])
        policies = np.frombuffer(p[1], np.float32).reshape(p[0])
        return cls(states.shape[0], states, values, policies)


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
        port: int = 7878,
        max_size: int = 500_000,
        min_size: int = 10_000,
        cache_dir: str = None,
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.replay_buffer = PriorityQueue()
        if cache_dir is not None and not cache_dir.endswith("/"):
            cache_dir += "/"
        self.cache_dir = cache_dir
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(f"tcp://*:{port}")
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, 1)

    def get_batch(self, batch_size: int = 32, refill=True) -> TrainingBatch:
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
        for _ in range(batch_size):
            data = self.replay_buffer.get()
            if data is None:
                break
            states.append(data.state)
            values.append(data.value)
            policies.append(data.policy)
            if refill:
                data.priority += HIGH_PRIORITY  # put at the end of the queue
                self.replay_buffer.put(data)

        states = np.stack(states)
        values = np.stack(values)
        policies = np.stack(policies)

        batch = TrainingBatch(batch_size, states, values, policies)
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

        while not self.replay_buffer.empty():
            batch = self.get_batch(1024, refill=False)
            fp = self.cache_dir + str(uuid.uuid4()) + ".npz"
            np.savez_compressed(
                fp, states=batch.states, values=batch.values, policies=batch.policies
            )

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
                    batch_size = int(msgpack.unpackb(message))
                    batch = self.get_batch(batch_size)
                    if batch is not None:
                        self.socket.send_multipart([address, b"", batch.pack()])
                    else:
                        self.socket.send_multipart([address, b"", b""])

                else:
                    self.process_data(message)
            except zmq.error.Again:
                pass
            except KeyboardInterrupt:
                break
        self.dump_to_disk()
        self.socket.close()
        self.context.term()
        print("DataBroker closed")


class DataGameClient:
    """Client for a GameGenerator to interact with the Databroker"""

    def __init__(self, port: int = 7878):
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://localhost:{port}")
        self.socket.setsockopt(zmq.IDENTITY, str(uuid.uuid4()).encode("utf-8"))

    def deposit(self, example: TrainState):
        """Deposits a single training example into the data broker"""
        self.socket.send(example.pack())

    def destroy(self):
        """Sanely shuts down the zmq client"""
        self.socket.close()
        self.context.term()
        print("DataTrainClient closed")


class DataTrainClient:
    """Client for a Trainer to interact with the Databroker"""

    def __init__(self, port: int = 7878, batch_size: int = 32):
        self.batch_size = batch_size
        identity = f"TRAIN-{uuid.uuid4()}".encode()
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.IDENTITY, identity)
        self.socket.connect(f"tcp://localhost:{port}")

    def get_batch(
        self,
    ) -> Union[TrainingBatch, None]:
        """Returns a TrainingBatch from the data broker"""
        self.socket.send(msgpack.packb(self.batch_size))
        return TrainingBatch.unpack(self.socket.recv())

    def destroy(self):
        """Sanely shuts down the zmq client"""
        self.socket.close()
        self.context.term()
        print("DataTrainClient closed")
