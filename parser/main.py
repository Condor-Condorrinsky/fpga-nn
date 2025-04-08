from torchvision.models import resnet50, ResNet50_Weights

if __name__ == '__main__':
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # print(model)

    params = []
    for param in model.parameters():
        # print(param)
        params.append(param)

    print(params)

    with open('test.txt', 'w') as f:
        for param in params:
            for weight in param.data:
                f.write(str(weight) + ',')
            f.write('\n')
