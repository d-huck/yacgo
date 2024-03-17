"""
Defines the inference and models for inference and training. These are simple 
wrappers around the EfficientFormer and gives extra methods for reloading and 
saving models, as well as interfacing with the ZMQ.
"""

import uuid
from typing import Tuple

import numpy as np
import torch
import zmq

from yacgo.utils import (
    pack_inference,
    unpack_examples,
    pack_state,
    unpack_inference,
    unpack_state,
)
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
        return f"ViTWrapper(model_size={self.model_size}, device={self.device}, board_size={self.board_size}, n_chans={self.n_chans})"

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

        TODO: Add a way to stop the server
        """
        while True:
            try:
                self.loop()
            except KeyboardInterrupt:
                break
        print("Job over, destroying sockets...")
        self.context.destroy()


class Trainer(ViTWrapper):  # TODO: implement training
    """
    Implements a trainer thread to continuously take in game states and train the model.
    """

    def __init__(self, port: int, args: dict, server_address: str = "localhost"):
        super().__init__(args)
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.IDENTITY, b"TRAIN-" + uuid.uuid4().bytes)
        self.socket.connect(f"tcp://{server_address}:{port}")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.regressor = torch.nn.MSELoss()
        self.dataset = None  # TODO: implement interfacing with dataset. This is simply a place holder.
        self.batch_size = args.training_batch_size
        self.training_steps = 500_000  #
        self.model.train()

    def get_batch(self):
        """
        Get a batch from the dataset

        Returns:
            _type_: _description_
        """
        return

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
        states = states.to(self.device)
        values = values.to(self.device)
        policies = policies.to(self.device)
        self.optimizer.zero_grad()
        values_pred, policy_pred = self.model.forward(states)
        loss = 0.5 * self.criterion(
            policy_pred.squeeze(), policies
        ) + 0.5 * self.regressor(values_pred.squeeze(), values)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(
        self,
    ):
        losses = []
        for i in range(self.training_steps):
            self.socket.send(self.batch_size)
            states, values, policies = unpack_examples(self.socket.recv())
            loss = self.train_step(states, policies, values)
            losses.append(loss)
            avg = sum(losses) / len(losses)
            print(f"Training Step {i}, Loss : {avg}")
            if len(losses) > 10:
                _ = losses.pop(0)


# if __name__ == "__main__":
#     print("This is the models module. Please run the main.py file.")

#     from yacgo.utils import make_args
#     from tqdm.auto import tqdm
#     import os
#     from datasets import Dataset
#     from torch.utils.data import DataLoader

#     args = make_args()
#     args.device = "cuda" if torch.cuda.is_available() else "mps"
#     args.num_feature_channels = 22
#     trainer = Trainer(args)
#     print(trainer)

#     data_dir = os.path.join(os.getcwd(), "data")

#     def generate_data(data_dir):
#         for root, dirs, files in os.walk(data_dir):
#             pos_len = 19
#             for file in files:
#                 if file.endswith(".npz"):
#                     npz = np.load(os.path.join(root, file))
#                     binaryInputNCHWPacked = npz["binaryInputNCHWPacked"]
#                     binaryInputNCHW = np.unpackbits(binaryInputNCHWPacked, axis=2)
#                     assert len(binaryInputNCHW.shape) == 3
#                     assert (
#                         binaryInputNCHW.shape[2] == ((pos_len * pos_len + 7) // 8) * 8
#                     )
#                     binaryInputNCHW = binaryInputNCHW[:, :, : pos_len * pos_len]
#                     valueTargetsNCHW = npz["valueTargetsNCHW"].astype(np.float32)
#                     policyTargetsNCMove = npz["policyTargetsNCMove"].astype(np.float32)
#                     globalTargetsNC = npz["globalTargetsNC"]
#                     scoreDistrN = npz["scoreDistrN"].astype(np.float32)

#                     def value_(x):
#                         if x[0] > 0:
#                             return 1.0
#                         elif x[1] > 0:
#                             return -1.0
#                         else:
#                             return 0.0

#                     policy = [p[0] for p in policyTargetsNCMove]
#                     values = [value_(p) for p in globalTargetsNC]
#                     states = np.reshape(
#                         binaryInputNCHW,
#                         (
#                             binaryInputNCHW.shape[0],
#                             binaryInputNCHW.shape[1],
#                             pos_len,
#                             pos_len,
#                         ),
#                     ).astype(np.float32)
#                     out = [
#                         {
#                             "state": s,
#                             "value": v,
#                             "policy": p,
#                         }
#                         for s, v, p in zip(states, values, policy)
#                     ]
#                     yield out

#     # examples = []

#     # for i, data in tqdm(enumerate(generate_data(data_dir))):
#     #     examples.extend(data)
#     #     if i > 1000:
#     #         break
#     # ds = Dataset.from_list(examples)

#     # ds.save_to_disk("data/valid_set")
#     print("Loading Data...")
#     dataset = Dataset.load_from_disk("data/valid_set")

#     print("Data Loaded")
#     device = "cuda" if torch.cuda.is_available() else "mps"
#     print("Device:", device)

#     print("Initializing Model...")
#     model = EfficientFormerV2(
#         depths=[2, 2, 6, 4],  # TODO: fine tune and make constants in model.py
#         in_chans=22,  # num of game state channels
#         img_size=19,  # TODO: args.board_size
#         embed_dims=(32, 64, 96, 172),
#         downsamples=(False, True, True, True),
#         num_vit=2,
#         mlp_ratios=(4, 4, (4, 3, 3, 3, 4, 4), (4, 3, 3, 4)),
#         num_classes=19**2 + 1,
#     ).to(device)

#     def collate_fn(example):
#         state = torch.stack([torch.tensor(ex["state"]) for ex in example])
#         value = torch.stack([torch.tensor(ex["value"]) for ex in example])
#         policy = torch.stack([torch.tensor(ex["policy"]) for ex in example])

#         return state, value, policy

#     dataloader = DataLoader(
#         dataset,
#         batch_size=256,
#         shuffle=True,
#         num_workers=8,
#         pin_memory=True,
#         collate_fn=collate_fn,
#     )

#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#     criterion = torch.nn.CrossEntropyLoss()
#     regressor = torch.nn.MSELoss()

#     model.train()
#     print("Beginning training...")
#     pbar = tqdm(total=500)
#     for _ in range(500):
#         loss_avg = 0
#         for i, (states, values, policies) in tqdm(
#             enumerate(dataloader), total=len(dataloader), leave=False
#         ):
#             states = states.to(device)
#             values = values.to(device)
#             policies = policies.to(device)
#             optimizer.zero_grad()
#             values_pred, policy_pred = model.forward(states)
#             loss = 0.5 * criterion(policy_pred.squeeze(), policies) + 0.5 * regressor(
#                 values_pred.squeeze(), values
#             )
#             loss.backward()
#             loss_avg += loss.detach().item()
#             optimizer.step()

#             if i % 10 == 0 and i != 0:
#                 loss = loss_avg / 25
#                 loss_avg = 0
#                 tqdm.write(f"Average Batch Loss: {loss}")
#         pbar.update(1)
