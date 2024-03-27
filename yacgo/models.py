"""
Defines the inference and models for inference and training. These are simple 
wrappers around the EfficientFormer and gives extra methods for reloading and 
saving models, as well as interfacing with the ZMQ.
"""

import atexit
import os
import random
import time
from typing import Tuple

import numpy as np
import torch
import wandb
import zmq
from tqdm.auto import tqdm

from yacgo.data import DATA_DTYPE, DataTrainClientMixin, Inference, State, TrainState
from yacgo.go import game
from yacgo.vit import (
    EfficientFormer_depth,
    EfficientFormer_expansion_ratios,
    EfficientFormer_width,
    EfficientFormerV2,
)

REQUEST_TIMEOUT = 5000  # ms


class Model(object):
    """
    Basic API for model interaction.
    """

    def __call__(self, inputs: TrainState) -> Tuple[np.float32, np.ndarray]:
        self.forward(inputs)

    def forward(self, inputs: TrainState) -> Tuple[np.float32, np.ndarray]:
        """Abstract class for unified forward method

        Args:
            inputs (np.ndarray): single state of a game

        Raises:
            NotImplementedError: must be implemented by inheritor
        """
        raise NotImplementedError()


class ViTWrapper(object):
    """
    Simple wrapper around EfficientFormerV2 to keep the implementation as close
    to the original as possible. Handles construction, saving and loading of weights.
    """

    def __init__(self, args: dict, model_path=None):
        depths = EfficientFormer_depth[args.model_size]
        embed_dims = EfficientFormer_width[args.model_size]
        mlp_ratios = EfficientFormer_expansion_ratios[args.model_size]

        self.device = args.device
        self.model = EfficientFormerV2(
            depths=depths,
            in_chans=args.num_feature_channels,
            img_size=args.board_size,
            embed_dims=embed_dims,
            num_vit=2,
            mlp_ratios=mlp_ratios,
            num_classes=args.board_size**2 + 1,
        ).to(self.device)
        self.model_size = args.model_size
        if model_path is not None:
            self.load_pretrained(model_path)
        elif args.model_path is not None:
            self.load_pretrained(args.model_path)
        self.board_size = args.board_size
        self.n_chans = args.num_feature_channels

    def __repr__(self):
        return f"ViTWrapper(model_size={self.model_size}, device={self.device}, board_size={self.board_size}, n_chans={self.n_chans})"  # pylint: disable=line-too-long

    def load_pretrained(self, path: str):
        """
        Load pretrained model. Provided model weights may be for a smaller
        version of the model, in which layers not present in the weights should
        be initialized.

        TODO: please verify this method
        Args:
            path (str): path to the weights

        Raises:
            NotImplementedError: _description_
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def save_pretrained(
        self, path: str = None, epoch: str = None, candidate: bool = False
    ):
        """Save pretrained weights to disk

        Args:
            path (str, optional): directory to store weights. Defaults to None.
            epoch (str, optional): epoch of training. Defaults to None.
            candidate (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        if path is None:
            path = "models/"
        if epoch is None:
            epoch = 0
        os.makedirs(path, exist_ok=True)
        if candidate:
            model_name = "candidate_model.pth"
        else:
            model_name = f"{self.model_size}-bs{self.board_size}-nc{self.n_chans}-epoch-{epoch:03d}.pth"
        out = os.path.join(path, model_name)
        torch.save(self.model.state_dict(), out)

        return out


class InferenceRandom(Model):
    """Simple Model interface that returns random values for playing games against"""

    def __init__(self):
        pass

    def forward(self, inputs):
        return DATA_DTYPE(np.random.random() * 2 - 1), np.random.random(
            game.action_size(inputs)
        ).astype(DATA_DTYPE)


class InferenceEqual(Model):
    """Simple Model interface that returns equal values for playing games against."""

    def __init__(self):
        pass

    def forward(self, inputs):
        pol = np.zeros(game.action_size(inputs)) + (1 / game.action_size(inputs))
        return 0, pol


class InferenceLocal(ViTWrapper, Model):
    """Simple Model interface that handles inference locally, for playing games against
    the bot or simple testing.
    """

    def __init__(self, args: dict):
        super().__init__(args)
        self.model.eval()

    def forward(self, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass of the model

        Args:
            inputs (np.ndarray): state of single game

        """
        inputs = torch.tensor(inputs).to(self.device)
        inputs = inputs.unsqueeze(dim=0)  # pretend there is a batch of size 1
        value, policy = self.model.forward(inputs)
        value = value.detach().cpu().numpy().squeeze()
        policy = policy.detach().cpu().numpy().squeeze()
        return value, policy


class InferenceClient(Model):
    """Client for inference server. Implements a simple Model API but redirects inferences
    to any available inference server.
    """

    def __init__(self, ports: list, server_address: str = "localhost"):
        self.context = zmq.Context.instance()
        self.server_address = server_address
        self.ports = ports

        def close():
            self.context.destroy()

        atexit.register(close)

    def try_request(self, req):
        """Reliable pattern for sending requests to the server. Will retry on failure indefinitely"""
        random.shuffle(self.ports)
        for port in self.ports:
            server = f"tcp://{self.server_address}:{port}"
            client = self.context.socket(zmq.REQ)
            client.setsockopt(zmq.LINGER, 0)
            client.connect(server)
            client.send(req)
            poll = zmq.Poller()
            poll.register(client, zmq.POLLIN)
            socks = dict(poll.poll(REQUEST_TIMEOUT))
            if socks.get(client) == zmq.POLLIN:
                reply = client.recv()
            else:
                reply = None
            poll.unregister(client)
            client.close()

            return reply

    def forward(self, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """_summary_

        Args:
            inputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        state = State(inputs).pack()
        inf = None
        while inf is None:
            inf = self.try_request(state)
        inf = Inference.unpack(inf)

        return inf.value, inf.policy


class InferenceServer(ViTWrapper):
    """
    Class for Inference server. Handles loading a model

    Args:
        ViTWrapper (_type_): _description_
    """

    def __init__(self, port, args: dict, model_path=None):
        super().__init__(args, model_path)
        self.model.eval()
        self.port = port
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(f"tcp://*:{self.port}")
        self.socket.setsockopt(zmq.LINGER, 0)

        self.socket.setsockopt(zmq.RCVTIMEO, 0)

        self.batch_size = args.inference_batch_size
        self.n_chans = args.num_feature_channels
        self.board_size = args.board_size

        def close():
            self.context.destroy()

        atexit.register(close)

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
        """
        main loop for the server. Expects to receive a batch of inputs and
        returns inference to the appropriate address.
        """
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
                state = State.unpack(buffer)
            except (zmq.error.Again, ValueError):
                continue
            inputs[n] = state.state
            addresses.append(address)
            n += 1
        # only respond if there is someone to return to. Ignores the case
        # where there are no games going on.
        if n > 0:
            inputs = np.copy(inputs[:n])
            value, policy = self.inference(inputs)
            for i, address in enumerate(addresses):
                inf = Inference(value[i], policy[i])
                self.socket.send_multipart([address, b"", inf.pack()])

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
        self.training_steps = args.training_steps_per_epoch
        self.wandb = args.wandb
        self.model.train()

    def train_step(self, states: np.ndarray, policies: np.ndarray, values: np.ndarray):
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
        if self.wandb:
            wandb.log({"Loss": loss.detach().cpu().item()})
        return loss.detach().cpu().item()

    def run(self, epoch: int = 0):
        """
        Runs the trainer indefinitely. Expects to receive a batch of inputs and
        """
        losses = []
        pbar = tqdm(
            desc=f"Training epoch {epoch}",
            total=self.training_steps,
            leave=False,
            smoothing=0.01,
        )
        for _ in range(self.training_steps):
            batch = self.get_batch()
            while batch.batch_size == 0:
                time.sleep(5)
                batch = self.get_batch()

            loss = self.train_step(batch.states, batch.policies, batch.values)
            losses.append(loss)
            avg = sum(losses) / len(losses)
            pbar.update(1)
            pbar.set_postfix({"Loss": f"{avg:04.4f}"})
            if len(losses) > 10:
                _ = losses.pop(0)


class ModelServerMixin:
    """Model Server to distribute weights of the most up to date model"""

    def __init__(self, args):
        self.address = (
            "*" if args.model_server_address is None else args.model_server_address
        )
        self.port = args.model_server_port
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{self.port}")
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind(f"tcp://*:{self.port + 1}")

        def close():
            self.context.destroy()

        atexit.register(close)


class ModelClientMixin:
    """Model client to receive weights from the model server"""

    def __init__(self, address: str, port: int):
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{address}:{port}")
        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.connect(f"tcp://{address}:{port + 1}")

        def close():
            self.context.destroy()

        atexit.register(close)
