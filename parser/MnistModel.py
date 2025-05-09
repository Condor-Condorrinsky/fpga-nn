import torch

class MnistModel(torch.nn.Module):

    def __init__(self):
        super(MnistModel, self).__init__()

        # in_features - number of inputs to each neuron, out_features - number of neurons
        self.linear1 = torch.nn.Linear(784, 16)
        self.activation1 = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(16, 16)
        self.activation2 = torch.nn.Sigmoid()
        self.linear3 = torch.nn.Linear(16, 10)
        self.activation3 = torch.nn.Sigmoid()

        torch.nn.init.uniform_(self.linear1.weight, -5, 5)
        torch.nn.init.uniform_(self.linear2.weight, -5, 5)
        torch.nn.init.uniform_(self.linear3.weight, -5, 5)

    def forward(self, x):
        x = self.activation1(self.linear1(x))
        x = self.activation2(self.linear2(x))
        x = self.activation3(self.linear3(x))
        return x

if __name__ == '__main__':
    import random
    test = [random.choice(range(256)) for _ in range(784)]
    m = MnistModel()
    out = m.forward(torch.tensor([test], dtype=torch.float32))
    print(out)
