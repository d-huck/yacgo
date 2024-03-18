"""
Defines the inference and models for inference and training. These are simple 
wrappers around the EfficientFormer and gives extra methods for reloading and 
saving models, as well as interfacing with the ZMQ.
"""

import time
import uuid
from typing import Tuple

import numpy as np
import torch
import zmq

from yacgo.data import DataTrainClientMixin
from yacgo.go import game
from yacgo.utils import pack_inference, pack_state, unpack_inference, unpack_state
from yacgo.vit import (
    EfficientFormer_depth,
    EfficientFormer_expansion_ratios,
    EfficientFormer_width,
    EfficientFormerV2,
)


class Model(object):
    """
    Basic API for model interaction.
    """

    def __call__(self, inputs):
        self.forward(inputs)

    def forward(self, inputs):
        """Abstract class for unified forward method

        Args:
            inputs (np.ndarray): single state of a game

        Raises:
            NotImplementedError: must be implemented by inheritor
        """
        raise NotImplementedError()


class ViTWrapper(object):
    """Simple wrapper around EfficientFormerV2 to keep the implementation as close
    to the original as possible. Handles construction, saving and loading of weights.
    """

    def __init__(self, args: dict):
        depths = EfficientFormer_depth[args.model_size]
        embed_dims = EfficientFormer_width[args.model_size]
        mlp_ratios = EfficientFormer_expansion_ratios[args.model_size]

        self.model = EfficientFormerV2(
            depths=depths,
            in_chans=args.num_feature_channels,
            img_size=args.board_size,
            embed_dims=embed_dims,
            num_vit=2,
            mlp_ratios=mlp_ratios,
            num_classes=args.board_size**2 + 1,
        ).to(args.device)
        self.model_size = args.model_size
        self.device = args.device
        self.board_size = args.board_size
        self.n_chans = args.num_feature_channels

    def __repr__(self):
        return f"ViTWrapper(model_size={self.model_size}, device={self.device}, board_size={self.board_size}, n_chans={self.n_chans})"  # pylint: disable=line-too-long

    def load_pretrained(self, path: str):
        """
        Load pretrained model. Provided model weights may be fore a smaller
        version of the model, in which layers not present in the weights should
        be initialized.

        TODO: please verify this method
        Args:
            path (str): path to the weights

        Raises:
            NotImplementedError: _description_
        """

        self.model.load_state_dict(torch.load(path))

    def save_pretrained(self, path: str):
        """
        Save pretrained weights.

        TODO: please verify this method
        Args:
            path (str): path to save the weights

        Raises:
            NotImplementedError: _description_
        """
        self.model.save_state_dict(path)


class InferenceRandom(Model):
    def __init__(self):
        pass

    def forward(self, inputs):
        return np.random.random(), np.random.random(game.action_size(inputs))


class InferenceEqual(Model):
    def __init__(self):
        pass

    def forward(self, inputs):
        pol = np.zeros(game.action_size(inputs)) + (1 / game.action_size(inputs))
        return 0, pol


class InferenceLocal(Model, ViTWrapper):
    """Simple Model interface that handles inference locally, for playing games against
    the bot or simple testing.
    """

    def __init__(self, args: dict):
        ViTWrapper.__init__(self, args)

    def forward(self, inputs):
        """Forward pass of the model

        Args:
            inputs (np.ndarray): state of single game

        """
        return self.model.forward(inputs)


class InferenceClient(Model):
    """Client for inference server. Implements a simple Model API but redirects inferences
    to any available inference server.
    """

    def __init__(self, ports: list, server_address: str = "localhost"):
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.IDENTITY, uuid.uuid4().bytes)
        for port in ports:
            self.socket.connect(f"tcp://{server_address}:{port}")

    def forward(self, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """_summary_

        Args:
            inputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.socket.send(pack_state(inputs))
        return unpack_inference(self.socket.recv())


class InferenceServer(ViTWrapper):
    """
    Class for Inference server. Handles loading a model

    Args:
        ViTWrapper (_type_): _description_
    """

    def __init__(self, port: int, args: dict):
        super().__init__(args)

        self.model.eval()
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(f"tcp://*:{port}")
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, 1)  # TODO: compare with immediate time out

        self.batch_size = args.inference_batch_size
        self.n_chans = args.num_feature_channels
        self.board_size = args.board_size

    def inference(self, inputs: np.ndarray):
        """Performs a batch inference on the model

        Args:
            inputs (np.ndarray): inputs of shape (batch_size, num_feature_channels,
            board_size, board_size)

        Returns:
            _type_: _description_
        """
        inputs = torch.Tensor(inputs).to(self.device)
        with torch.no_grad():
            outputs = self.model.forward(inputs)
        value = outputs[0].cpu().numpy()
        policy = outputs[1].cpu().numpy()
        return value, policy

    def loop(self):
        addresses = []
        inputs = np.empty(
            (
                self.batch_size,
                self.n_chans,
                self.board_size,
                self.board_size,
            )
        )

        n = 0
        for _ in range(self.batch_size):
            try:
                address, _, buffer = self.socket.recv_multipart()
            except zmq.error.Again:
                continue
            inputs[n] = unpack_state(buffer)
            addresses.append(address)
            n += 1
        # only respond if there is someone to return to. Ignores the case
        # where there are no games going on.
        if n > 0:
            inputs = np.copy(inputs[:n])
            value, policy = self.inference(inputs)
            for i, address in enumerate(addresses):
                self.socket.send_multipart(
                    [address, b"", pack_inference(value[i], policy[i])]
                )

    def run(self):
        """
        Runs the server indefinitely. Expects to receive a batch of inputs and
        returns inference to the appropriate address.
        """
        while True:
            try:
                self.loop()
            except KeyboardInterrupt:
                break
        print("Job over, destroying sockets...")
        self.context.destroy()


class Trainer(ViTWrapper, DataTrainClientMixin):
    """
    Implements a trainer thread to continuously take in game states and train the model.
    """

    def __init__(self, args: dict):
        super(Trainer, self).__init__(args)
        DataTrainClientMixin.__init__(self, args)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.regressor = torch.nn.MSELoss()
        self.training_steps = 500_000  #
        self.model.train()

    def train_step(
        self, states: torch.Tensor, policies: torch.Tensor, values: torch.Tensor
    ):
        """
        Perform a single training step on the model.

        Args:
            states (torch.Tensor): input game states, must match model size
            policies (torch.Tensor): target polices for examples
            values (torch.Tensor): target values for examples

        Returns:
            torch.float32: loss of the training step
        """
        states = torch.tensor(states).to(self.device)
        values = torch.tensor(values).to(self.device)
        policies = torch.tensor(policies).to(self.device)
        self.optimizer.zero_grad()
        values_pred, policy_pred = self.model.forward(states)
        loss = 0.5 * self.criterion(
            policy_pred.squeeze(), policies
        ) + 0.5 * self.regressor(values_pred.squeeze(), values)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def run(
        self,
    ):
        """
        Runs the trainer indefinitely. Expects to receive a batch of inputs and
        """
        losses = []
        try:
            for i in range(self.training_steps):
                batch = self.get_batch()
                if batch.batch_size == 0:
                    print("Empty Batch, sleeping...")
                    time.sleep(5)
                    continue
                loss = self.train_step(batch.states, batch.policies, batch.values)
                losses.append(loss)
                avg = sum(losses) / len(losses)
                print(f"Training Step {i:06,d}, Loss : {avg:04.4f}", end="\r")
                if len(losses) > 10:
                    _ = losses.pop(0)
        except KeyboardInterrupt:
            pass
        self.destroy()
        print("Trainer has quit")
