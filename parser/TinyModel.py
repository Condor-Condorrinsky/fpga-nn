import torch

class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        # in_features - number of inputs to each neuron, out_features - number of neurons
        self.linear1 = torch.nn.Linear(10, 10)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(10, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
