import torch

def dump_model(model: torch.nn.Module, outfile: str):
    i = 0
    j = 0
    with open(outfile, "w") as f:
        for layer in model.children():
            if isinstance(layer, torch.nn.Linear):
                f.write(f"Layer nr {i}:\n")
                for neuron in layer.state_dict()['weight']:
                    f.write(f"Neuron nr {j}:")
                    for neuron_weight in neuron:
                        f.write(f"{neuron_weight.item()},")
                    j += 1
                    f.write("\n")
                j = 0
                f.write(f"Biases nr {i}:")
                for bias in layer.state_dict()['bias']:
                    f.write(f"{bias.item()},")
                f.write("\n")
            i += 1
