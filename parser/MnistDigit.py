
def load_mnist_csv(path: str):
    ret = []
    with open(path, 'r') as f:
        for line in f:
            nums = [int(x) for x in line.split(',')]
            ret.append(MnistDigit(nums[0], nums[1:]))
    return ret

class MnistDigit:

    def __init__(self, label: int, pixels: list[int]):
        if label < 0 or label > 9:
            raise ValueError("Label must be between 0 and 9")
        if len(pixels) != 784:
            raise ValueError("Pixels list length must be equal to 784")

        self.label = label
        self.pixels = pixels
