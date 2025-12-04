import brevitas
import brevitas.nn
import torch


class QuantizedMnistModel(torch.nn.Module):

    def __init__(self):
        super(QuantizedMnistModel, self).__init__()

        # Quantizes input tensor, we cannot directly feed float Tensor here
        self.quant_input = brevitas.nn.QuantIdentity(bit_width=8)

        self.q_linear1 = brevitas.nn.QuantLinear(in_features=784, out_features=16, bias=True, weight_bit_width=8)
        self.q_activation1 = brevitas.nn.QuantSigmoid()
        self.q_linear2 = brevitas.nn.QuantLinear(in_features=16, out_features=16, bias=True, weight_bit_width=8)
        self.q_activation2 = brevitas.nn.QuantSigmoid()
        self.q_linear3 = brevitas.nn.QuantLinear(in_features=16, out_features=10, bias=True, weight_bit_width=8)
        self.q_activation3 = brevitas.nn.QuantSigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant_input(x)
        x = self.q_linear1(x)
        x = self.q_activation1(x)
        x = self.q_linear2(x)
        x = self.q_activation2(x)
        x = self.q_linear3(x)
        x = self.q_activation3(x)
        return x
