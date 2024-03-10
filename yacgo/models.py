"""
Defines the inference and models for inference and training. These are simple 
wrappers around the EfficientFormer and gives extra methods for reloading and 
saving models, as well as interfacing with the ZMQ.
"""

import uuid

import msgpack
import numpy as np
import torch
import zmq
import torch

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

        TODO: please implement this method
        Args:
            path (str): path to the weights

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    def save_pretrained(self, path: str):
        """
        Save pretrained weights.

        TODO: please implement this method
        Args:
            path (str): path to save the weights

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()


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

    def pack(self, inputs):
        """Pack input data into a byte array for transmission.

        Args:
            inputs (np.ndarray): state of a current game

        Returns:
            bytes: array for transmission
        """
        return msgpack.packb((inputs.shape, str(inputs.dtype), inputs.tobytes()))

    def unpack(self, buffer):
        """Unpack data from the server.

        Args:
            buffer (_type_): _description_

        Returns:
            _type_: _description_
        """
        policy_shape, dtype, value_buffer, policy_buffer = msgpack.unpackb(buffer)
        value = np.frombuffer(value_buffer, dtype)
        policy = np.frombuffer(policy_buffer, dtype).reshape(policy_shape)
        return value, policy

    def forward(self, inputs):
        """_summary_

        Args:
            inputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.socket.send(self.pack(inputs))
        return self.unpack(self.socket.recv())


class InferenceServer(ViTWrapper):
    """
    Class for Inference server. Handles loading a model

    Args:
        ViTWrapper (_type_): _description_
    """

    def __init__(self, args: dict, port: int):
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

    def unpack(self, buffer):
        """Unpacks information from the buffer into the array. Expects all incoming data
        to be of the same dimension of the model inputs.

        Args:
            buffer (bytes): buffer packed by InferenceClient

        Returns:
            np.ndarray: Single example for inference
        """
        shape, dtype, value = msgpack.unpackb(buffer)
        return np.frombuffer(value, dtype).reshape(shape)

    def pack(self, value, policy):
        """Packs the value and policy into a buffer to be sent back to a client.

        Args:
            value (np.float32): single value score between -1 and 1
            policy (np.ndarray): policy of shape board_size ** 2 + 1

        Returns:
            bytes: buffer to be sent back to the client
        """
        return msgpack.packb(
            (
                policy.shape,
                str(value.dtype),
                value.tobytes(),
                policy.tobytes(),
            )
        )

    def run(self):
        """
        Runs the server indefinitely. Expects to receive a batch of inputs and returns inference
        to the appropriate address.

        TODO: Add a way to stop the server
        """
        while True:
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
                inputs[n] = self.unpack(buffer)
                addresses.append(address)
                n += 1
            # only respond if there is someone to return to. Ignores the case
            # where there are no games going on.
            if n > 0:
                inputs = np.copy(inputs[:n])
                value, policy = self.inference(inputs)
                for i, address in enumerate(addresses):
                    self.socket.send_multipart(
                        [address, b"", self.pack(value[i], policy[i])]
                    )


class Trainer(ViTWrapper):  # TODO: implement training
    """
    Implements a trainer thread to continuously take in game states and train the model.
    """

    def __init__(self, args: dict):
        super().__init__(args)


if __name__ == "__main__":
    print("This is the models module. Please run the main.py file.")

    from yacgo.utils import make_args
    from tqdm.auto import tqdm
    import os
    from torch.utils.data import TensorDataset, DataLoader

    args = make_args()
    args.device = "cuda" if torch.cuda.is_available() else "mps"
    args.num_feature_channels = 22
    trainer = Trainer(args)
    print(trainer)

    data_dir = os.path.join(os.getcwd(), "data")

    def generate_data(data_dir):
        for root, dirs, files in os.walk(data_dir):
            pos_len = 19
            for file in files:
                if file.endswith(".npz"):
                    npz = np.load(os.path.join(root, file))
                    binaryInputNCHWPacked = npz["binaryInputNCHWPacked"]
                    binaryInputNCHW = np.unpackbits(binaryInputNCHWPacked, axis=2)
                    assert len(binaryInputNCHW.shape) == 3
                    assert (
                        binaryInputNCHW.shape[2] == ((pos_len * pos_len + 7) // 8) * 8
                    )
                    binaryInputNCHW = binaryInputNCHW[:, :, : pos_len * pos_len]
                    valueTargetsNCHW = npz["valueTargetsNCHW"].astype(np.float32)
                    policyTargetsNCMove = npz["policyTargetsNCMove"].astype(np.float32)
                    globalTargetsNC = npz["globalTargetsNC"]
                    policy = np.stack([p[0] for p in policyTargetsNCMove])
                    scoreDistrN = npz["scoreDistrN"].astype(np.float32)

                    def value_(x):
                        if x[0] > 0:
                            return 1.0
                        elif x[1] > 0:
                            return -1.0
                        else:
                            return 0.0

                    values = np.stack([value_(p) for p in globalTargetsNC])
                    state = np.reshape(
                        binaryInputNCHW,
                        (
                            binaryInputNCHW.shape[0],
                            binaryInputNCHW.shape[1],
                            pos_len,
                            pos_len,
                        ),
                    ).astype(np.float32)
                    out = {
                        "state": state,
                        "values": values,
                        "policy": policy,
                    }
                    yield out

    states = []
    values = []
    policies = []
    for i, data in enumerate(generate_data(data_dir)):
        states.append(data["state"])
        values.append(data["values"])
        policies.append(data["policy"])

    print("Loading Data...")
    states = np.concatenate(states, axis=0).tolist()
    values = np.concatenate(values, axis=0).tolist()
    policies = np.concatenate(policies, axis=0).tolist()

    print("Data Loaded")
    device = "cuda" if torch.cuda.is_available() else "mps"
    print("Device:", device)

    print("Initializing Model...")
    model = EfficientFormerV2(
        depths=[2, 2, 6, 4],  # TODO: fine tune and make constants in model.py
        in_chans=22,  # num of game state channels
        img_size=19,  # TODO: args.board_size
        embed_dims=(32, 64, 96, 172),
        downsamples=(False, True, True, True),
        num_vit=2,
        mlp_ratios=(4, 4, (4, 3, 3, 3, 4, 4), (4, 3, 3, 4)),
        num_classes=19**2 + 1,
    ).to(device)

    states = torch.Tensor(states)
    policies = torch.Tensor(policies)
    values = torch.Tensor(values)

    print("Creating Dataset...")
    dataset = TensorDataset(states, values, policies)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    regressor = torch.nn.MSELoss()

    model.train()
    print("Beginning training...")
    for i, (state, value, policy) in tqdm(enumerate(dataloader)):
        state = state.to(device)
        value = value.to(device)
        policy = policy.to(device)
        optimizer.zero_grad()
        out = model.forward(state)
        loss = 0.5 * criterion(out[1], policy) + 0.5 * regressor(out[0], value)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Loss: {loss.item()}")
        if i > 1000:
            break
