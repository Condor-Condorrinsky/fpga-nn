import torch

from MnistModel import MnistModel

def quantize_model(model_path: str, output_path: str):
    model = MnistModel()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    with open(output_path, "w") as f:
        for layer, value in model._modules.items():
            if isinstance(value, torch.nn.Linear):
                # for i in range(0, 16):
                for i in [x / 100.0 for x in range(-100, 105, 5)]:
                    num_of_features = value.in_features
                    test_tensor = torch.tensor([float(i)] * num_of_features, dtype=torch.float32)
                    out = model.forward_one_layer(test_tensor, layer)
                    f.write(f"Layer {layer} quantized for i={i}:\n")
                    for val in out:
                        f.write(f"{val},")
                    f.write("\n")
