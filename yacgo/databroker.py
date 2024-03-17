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

import msgpack
import numpy as np
import zmq

from yacgo.utils import pack_examples, unpack_examples, DATA_DTYPE

DEPOSIT = 0
TRAINING_BATCH = 1
QUIT = -1


@dataclass(order=True)
class PrioritizedData(object):
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


class DataBroker(object):
    """
    Databroker class to handle the consolidation and serving of data to the
    Trainer.

    Args:
        object (_type_): _description_
    """

    def __init__(self, port: int = 7878, max_size: int = 500_000):
        self.max_size = max_size
        self.replay_buffer = PriorityQueue()
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(f"tcp://*:{port}")
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, 1)

    def get_batch(self, batch_size: int = 32):
        """Returns a single batch from the replay buffer

        Args:
            batch_size (int, optional): _description_. Defaults to 32.

        Returns:
            _type_: _description_
        """
        states = []
        values = []
        policies = []
        for _ in range(batch_size):
            data = self.replay_buffer.get()
            states.append(data.state)
            values.append(data.value)
            policies.append(data.policy)
            data.priority += 1000  # put at the end of the queue
            self.replay_buffer.put(data)

        states = np.stack(states)
        values = np.stack(values)
        policies = np.stack(policies)

        return pack_examples(states, values, policies)

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
        states, values, policies = unpack_examples(message)
        assert (
            len(states.shape) == 4
            and len(values.shape) == 1
            and len(policies.shape) == 2
        ), "Invalid shapes for state, value, or policy"

        states = states.tolist()
        values = values.tolist()
        policies = policies.tolist()

        for s, v, p in zip(states, values, policies):
            data = PrioritizedData(randint(0, 100), s, v, p)
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
