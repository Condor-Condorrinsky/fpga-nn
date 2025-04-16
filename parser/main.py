from TinyModel import TinyModel

from torchvision.models import resnet50, ResNet50_Weights
import torch.nn

if __name__ == '__main__':
    # model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # # print(model)
    #
    # params = []
    # for param in model.parameters():
    #     # print(param)
    #     params.append(param)
    #
    # print(params)
    #
    # with open('test.txt', 'w') as f:
    #     for param in params:
    #         for weight in param.data:
    #             f.write(str(weight) + ',')
    #         f.write('\n')

    tinymodel = TinyModel()

    # print('The model:')
    # print(tinymodel)
    #
    # print('\n\nJust one layer:')
    # print(tinymodel.linear2)

    # print('\n\nModel params:')
    # for param in tinymodel.parameters():
    #     print(param)
    #
    # print('\n\nLayer params:')
    # for param in tinymodel.linear2.parameters():
    #     print(param)
    i = 0
    j = 0
    with open("tinytest.txt", "w") as f:
        for layer in tinymodel.children():
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
        i = 0



