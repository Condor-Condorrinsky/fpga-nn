from torchvision.transforms import v2

class Downscale(v2.Transform):

    __RGB_VALS = 256

    def __init__(self, divisor: int):
        super().__init__()

        assert divisor > 0 and Downscale.__RGB_VALS % divisor == 0,\
            "Divisor must be a positive integer that divides RGB_VALS"
        self.divisor = divisor

    def __call__(self, sample: dict):
        for i in range(len(sample['pixels'])):
            sample['pixels'][i] //= self.divisor
        return sample
