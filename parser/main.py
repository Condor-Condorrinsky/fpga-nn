from torchvision.models import resnet50, ResNet50_Weights

if __name__ == '__main__':
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # print(model)

    for param in model.parameters():
        print(param)
