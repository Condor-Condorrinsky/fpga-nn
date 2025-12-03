# === Ugly but whatever ===
# import sys
# import os.path
# sys.path.append(
#     os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from mnist.MnistModel import MnistModel
# =========================
from brevitas_quantization.QuantizedMnistModel import QuantizedMnistModel

# TODO: start QAT fine-tuning for MnistModel
import brevitas
import torch

def main():
    model = MnistModel()
    model.load_state_dict(torch.load("../models/eval_model/epoch_24"))
    quant_model = QuantizedMnistModel()
    quant_model.load_state_dict(model.state_dict(), strict=False)

if __name__ == '__main__':
    main()
