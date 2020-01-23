from torchvision import datasets, transforms
from jax.experimental.stax import (AvgPool, BatchNorm, Conv, Dense, FanInSum,
                                   FanOut, Flatten, GeneralConv, Identity,
                                   MaxPool, Relu, LogSoftmax, Dropout, Sigmoid, Tanh)
from utils import make_model
import torch
from jax.experimental import stax

def _input_shape(batch_size):
    return (batch_size, 1, 28, 28)

def get_mnist_dataset(batch_size):
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,)),
                           ])),
            batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                       ])),
        batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def LeNet5(batch_size, num_particles):
    input_shape = _input_shape(batch_size)
    return make_model(stax.serial(
        GeneralConv(('NCHW', 'OIHW', 'NHWC'), out_chan=6, filter_shape=(5, 5), strides=(1, 1), padding="VALID"),
        Relu,
        MaxPool(window_shape=(2, 2), strides=(2, 2), padding="VALID"),
        Conv(out_chan=16, filter_shape=(5, 5), strides=(1, 1), padding="SAME"),
        Relu,
        MaxPool(window_shape=(2, 2), strides=(2, 2), padding="SAME"),
        Conv(out_chan=120, filter_shape=(5, 5), strides=(1, 1), padding="VALID"),
        Relu,
        MaxPool(window_shape=(2, 2), strides=(2, 2), padding="SAME"),
        Flatten,
        Dense(84),
        Relu,
        Dense(10),
        LogSoftmax
    ), input_shape, num_particles)
