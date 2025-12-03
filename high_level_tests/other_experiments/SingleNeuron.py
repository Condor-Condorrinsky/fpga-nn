import math
import random
import typing

class SingleNeuron(object):

    def __init__(self, activation_f: typing.Callable) -> None:
        self.weight = random.uniform(-1, 1)
        self.bias = random.uniform(-1, 1)
        self.activation = activation_f

    def __str__(self):
        return f'Weight: {self.weight}, Bias: {self.bias}, Activation: {self.activation.__name__})'

    @classmethod
    def sigmoid(cls, x: int | float) -> float:
        return 1 / (1 + math.exp(-x))

    @classmethod
    def relu(cls, x: int | float) -> float:
        return max(0, x)

    def forward(self, x: int | float) -> float:
        return self.activation(self.weight * x + self.bias)


def main():
    random.seed(2025)
    nn = SingleNeuron(activation_f=SingleNeuron.relu)
    print(nn)
    for i in range(-100, 101, 1):
        x = i / 100
        print(f'x for i={i}: {nn.forward(x)}')


if __name__ == "__main__":
    main()
