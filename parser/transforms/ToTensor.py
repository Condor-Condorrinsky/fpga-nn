import torch

class ToTensor:

    def __call__(self, sample: dict):
        sample['pixels'] = torch.tensor(sample['pixels'])
        return sample
