import brevitas
import brevitas.nn
from brevitas.quant import Int8WeightPerTensorFixedPoint
import torch

import math

class QuantizedMnistModel(torch.nn.Module):

    def __init__(self, bit_width: int = 8):
        if bit_width < 1 or bit_width > 8:
            raise ValueError('Bit width must be between 1 and 8')

        super(QuantizedMnistModel, self).__init__()

        # Quantizes input tensor, we cannot directly feed float Tensor here
        self.quant_input = brevitas.nn.QuantIdentity(bit_width=bit_width)

        self.q_linear1 = brevitas.nn.QuantLinear(in_features=784,
                                                 out_features=16,
                                                 bias=True,
                                                 weight_bit_width=bit_width,
                                                 weight_quant=Int8WeightPerTensorFixedPoint)
        self.q_activation1 = brevitas.nn.QuantSigmoid()
        self.q_linear2 = brevitas.nn.QuantLinear(in_features=16,
                                                 out_features=16,
                                                 bias=True,
                                                 weight_bit_width=bit_width,
                                                 weight_quant=Int8WeightPerTensorFixedPoint)
        self.q_activation2 = brevitas.nn.QuantSigmoid()
        self.q_linear3 = brevitas.nn.QuantLinear(in_features=16,
                                                 out_features=10,
                                                 bias=True,
                                                 weight_bit_width=bit_width,
                                                 weight_quant=Int8WeightPerTensorFixedPoint)
        self.q_activation3 = brevitas.nn.QuantSigmoid()

        print(f"Weight fix point: {- math.log2(self.q_linear1.quant_weight().scale)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant_input(x)
        x = self.q_linear1(x)
        x = self.q_activation1(x)
        x = self.q_linear2(x)
        x = self.q_activation2(x)
        x = self.q_linear3(x)
        x = self.q_activation3(x)
        return x
