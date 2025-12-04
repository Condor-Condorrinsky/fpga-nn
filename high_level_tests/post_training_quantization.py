# === Ugly but whatever ===
# import sys
# import os.path
# sys.path.append(
#     os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from mnist.MnistModel import MnistModel
from mnist.MnistDataset import MnistDataset
from transforms.Downscale import Downscale
from transforms.ToTensor import ToTensor
# =========================
from brevitas_quantization.QuantizedMnistModel import QuantizedMnistModel
from training import train

# TODO: start QAT fine-tuning for MnistModel
import brevitas
import torch
import torchvision

def main():
    model = MnistModel()
    model.load_state_dict(torch.load("models/eval_model/epoch_24"))
    quant_model = QuantizedMnistModel()
    quant_model.load_state_dict(model.state_dict(), strict=False)

    transforms = torchvision.transforms.Compose([Downscale(16), ToTensor()])
    train_set = MnistDataset('mnist/MNIST_CSV/mnist_train.csv', transform=transforms)
    valid_set = MnistDataset('mnist/MNIST_CSV/mnist_test.csv', transform=transforms)

    train(
        training_set=train_set,
        validation_set=valid_set,
        batch_size=16,
        model=model,
        loss_function=torch.nn.MSELoss,
        learning_rate=1e-4,
        momentum=0.9,
        epochs=5,
    )

if __name__ == '__main__':
    main()
