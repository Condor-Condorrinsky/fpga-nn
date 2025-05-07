import torch

class MnistModel(torch.nn.Module):

    def __init__(self):
        super(MnistModel, self).__init__()

        # in_features - number of inputs to each neuron, out_features - number of neurons
        self.linear1 = torch.nn.Linear(1, 784)
        self.activation1 = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(784, 16)
        self.activation2 = torch.nn.Sigmoid()
        self.linear3 = torch.nn.Linear(16, 16)
        self.activation3 = torch.nn.Sigmoid()
        self.linear4 = torch.nn.Linear(16, 10)
        self.activation4 = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        x = self.activation3(x)
        x = self.linear4(x)
        x = self.activation4(x)
        return x
