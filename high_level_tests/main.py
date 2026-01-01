from mnist.MnistModel import MnistModel
from mnist.MnistDataset import MnistDataset
from transforms.Downscale import Downscale
from transforms.ToTensor import ToTensor
from primitive_quantize_model import quantize_model
from training import train
from brevitas_quantization.QuantizedMnistModel import QuantizedMnistModel
from utils.get_device import get_device

import torch
import torchvision

import sys


if __name__ == '__main__':
    dev = get_device()
    torch.set_default_device(dev)

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

    if sys.argv[1] == 'load-quantized':
        m = QuantizedMnistModel(bit_width=2)
        m.load_state_dict(torch.load('models/eval_quantized/epoch_4'), strict=False)
        print(f'Weight QuantTensor:\n {m.q_linear3.quant_weight()}')
