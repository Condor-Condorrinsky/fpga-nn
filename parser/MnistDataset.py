import torch

class MnistDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file: str, transform=None):
        self.data = []
        self.transform = transform

        self.__load_mnist_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __load_mnist_csv(self, path: str):
        self.data.clear()
        with open(path, 'r') as f:
            for line in f:
                nums = [int(x) for x in line.split(',')]
                self.data.append({'label': str(nums[0]), 'pixels': nums[1:]})
