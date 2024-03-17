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

from queue import PriorityQueue
from dataclasses import dataclass, field
from random import randint
import uuid

import msgpack
import numpy as np
import zmq

from yacgo.utils import pack_examples, unpack_examples, DATA_DTYPE
from yacgo.train_utils import TrainState

DEPOSIT = 0
TRAINING_BATCH = 1
QUIT = -1


@dataclass
class TrainingBatch:
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
        return cls(randint(0, 100), s.state.copy(), s.value.copy(), s.policy.copy())


class DataBroker(object):
    """
    Databroker class to handle the consolidation and serving of data to the
    Trainer.
    """

    def __init__(self, port: int = 7878, max_size: int = 500_000):
        self.max_size = max_size
        self.replay_buffer = PriorityQueue()
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(f"tcp://*:{port}")
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, 1)

    def get_batch(self, batch_size: int = 32) -> TrainingBatch:
        """Returns a single batch from the replay buffer

        Args:
            batch_size (int, optional): Batch size. Defaults to 32.

        Returns:
            TrainingBatch: Training Batch
        """
        states = []
        values = []
        policies = []
        for _ in range(batch_size):
            data = self.replay_buffer.get()
            states.append(data.state)
            values.append(data.value)
            policies.append(data.policy)
            data.priority += 100  # put at the end of the queue
            self.replay_buffer.put(data)

        states = np.stack(states)
        values = np.stack(values)
        policies = np.stack(policies)

        batch = TrainingBatch(batch_size, states, values, policies)
        return batch.pack()

    def load_from_disk(self, path):
        """
        Loads cached data from disk

        Args:
            path (_type_): _description_
        """

    def dump_to_disk(self, batch, path):
        """
        Dumps data to disk for caching

        Args:
            batch (_type_): _description_
            path (_type_): _description_
        """

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
                address = str(address.decode("utf-8"))
                if address.startswith("TRAIN"):
                    batch_size = int(msgpack.unpackb(message))
                    batch = self.get_batch(batch_size)
                    self.socket.send_multipart([address, b"", batch])
                else:
                    self.process_data(message)
            except zmq.error.Again:
                pass
            except KeyboardInterrupt:
                break
        self.socket.close()
        self.context.term()
        print("DataBroker closed")


class DataGameClient:
    """Client for a GameGenerator to interact with the Databroker"""

    def __init__(self, port: int = 7878):
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://localhost:{port}")
        self.socket.setsockopt(zmq.IDENTITY, uuid.uuid4().bytes)

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
        identity = f"TRAIN-{uuid.uuid4()}".encode("utf-8")
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://localhost:{port}")
        self.socket.setsockopt(zmq.IDENTITY, identity)

    def get_batch(
        self,
    ) -> TrainingBatch:
        """Returns a TrainingBatch from the data broker"""
        self.socket.send(msgpack.packb(self.batch_size))
        return TrainingBatch.unpack(self.socket.recv())

    def destroy(self):
        """Sanely shuts down the zmq client"""
        self.socket.close()
        self.context.term()
        print("DataTrainClient closed")
