from mnist.MnistModel import MnistModel
from mnist.MnistDataset import MnistDataset
from transforms.Downscale import Downscale
from transforms.ToTensor import ToTensor
from quantize_model import quantize_model
from training import train

import torch
import torchvision

import sys


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        transforms = torchvision.transforms.Compose([Downscale(16), ToTensor()])
        train_set = MnistDataset('mnist/MNIST_CSV/mnist_train.csv', transform=transforms)
        valid_set = MnistDataset('mnist/MNIST_CSV/mnist_test.csv', transform=transforms)
        mnist = MnistModel()

        train(
            training_set=train_set,
            validation_set=valid_set,
            batch_size=16,
            model=mnist,
            loss_function=torch.nn.MSELoss,
            learning_rate=0.001,
            momentum=0.9,
            epochs=25,
        )

    if sys.argv[1] == 'quantize':
        quantize_model(sys.argv[2], sys.argv[3])
