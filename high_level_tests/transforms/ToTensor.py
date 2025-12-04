import torch
from torchvision.transforms import v2

class ToTensor(v2.Transform):

    def __call__(self, sample: dict):
        super().__init__()

        sample['label'] = torch.tensor([sample['label']], dtype=torch.float32)
        sample['pixels'] = torch.tensor([sample['pixels']], dtype=torch.float32)
        return sample

if __name__ == '__main__':
    s = {'label_str': '1', 'label': [0, 1, 0], 'pixels': [1, 2, 3]}
    a = ToTensor()(s)
    print(a)
