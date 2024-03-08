"""
Testing multiple GPU servers with ZeroMQ. 
See https://stackoverflow.com/questions/73764403/how-can-a-zmq-server-process-requests-in-batches
"""

import argparse
import multiprocessing as mp
import time
import uuid

import msgpack
import numpy as np
import torch
import zmq

from yacgo.vit import EfficientFormerV2

JOBS = 1000


def pack(array):
    return msgpack.packb((array.shape, str(array.dtype), array.tobytes()))


def unpack(buffer):
    shape, dtype, value = msgpack.unpackb(buffer)
    return np.frombuffer(value, dtype).reshape(shape)


def computation(inputs: np.ndarray, model: EfficientFormerV2, args):
    inputs = torch.Tensor(inputs).to(args.device)  # args.device
    with torch.no_grad():
        outputs = model(inputs)
    return outputs[1].cpu().numpy()


def inference_server(port, args):
    context = zmq.Context.instance()
    socket = context.socket(zmq.ROUTER)
    socket.bind(f"tcp://*:{port}")

    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, 1)

    model = EfficientFormerV2(
        depths=[2, 2, 6, 4],  # TODO: fine tune and make constants in model.py
        in_chans=12,  # num of game state channels
        img_size=args.board_size,  # TODO: args.board_size
        embed_dims=(32, 64, 96, 172),
        downsamples=(False, True, True, True),
        num_vit=2,
        mlp_ratios=(4, 4, (4, 3, 3, 3, 4, 4), (4, 3, 3, 4)),
        num_classes=args.board_size**2 + 1,
    ).to(args.device)
    while True:
        inputs = np.empty(
            (
                args.batch_size,
                args.num_feature_channels,
                args.board_size,
                args.board_size,
            )
        )

        addresses = []

        n = 0
        for i in range(args.batch_size):
            try:
                address, empty, payload = socket.recv_multipart()
            except zmq.error.Again:
                continue

            inputs[n] = unpack(payload)
            addresses.append(address)
            n += 1

        # only return if there is someone to return to.
        # Ignores the case when no games are going on
        if n > 0:
            inputs = np.copy(inputs[:n])
            results = computation(inputs, model, args)

            for i, address in enumerate(addresses):
                if address is None:
                    continue
                payload = pack(inputs[i])
                socket.send_multipart([address, b"", payload])
            # print("Send response batch.", flush=True)


def client(ports, args):
    context = zmq.Context.instance()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.IDENTITY, uuid.uuid4().bytes)
    for port in ports:
        socket.connect(f"tcp://localhost:{port}")
    durations = []
    for _ in range(JOBS):
        input_ = np.random.randn(
            args.num_feature_channels, args.board_size, args.board_size
        ).astype(np.float32)
        start = time.time()
        socket.send(pack(input_))
        result = unpack(socket.recv())
        stop = time.time()
        durations.append(stop - start)
    print(f"Client average latency: {sum(durations)/len(durations):.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--board_size", type=int, default=19)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_servers", type=int, default=2)
    parser.add_argument("--num_games", type=int, default=128)
    parser.add_argument("--num_feature_channels", "-fc", type=int, default=12)

    args = parser.parse_args()

    ports = list(range(5550, 5550 + args.num_servers))
    games = []
    servers = []
    for port in ports:
        servers.append(mp.Process(target=inference_server, args=(port, args)))

    for _ in range(args.num_games):
        games.append(mp.Process(target=client, args=(ports, args)))

    for server in servers:
        server.start()

    start = time.time()
    for game in games:
        game.start()

    count = 0
    for game in games:
        game.join()
        # count += 1
        # # print(f"{count} games finished", end="\r")

    end = time.time()
    n_games = JOBS * args.num_games
    duration = end - start
    print(
        f"Evaluated {n_games} game states. Time taken: {duration:.2f} seconds or {n_games / duration:.4f} states per second.",
        flush=True,
    )

    for server in servers:
        server.terminate()

    print("exiting!")
