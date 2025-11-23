import random

from SingleNeuron import SingleNeuron


def main():
    random.seed(2025)
    nn = SingleNeuron(activation_f=SingleNeuron.relu)
    print(nn)
    for i in range(-100, 101, 1):
        x = i / 100
        print(f'x for i={i}: {nn.forward(x)}')


if __name__ == "__main__":
    main()
