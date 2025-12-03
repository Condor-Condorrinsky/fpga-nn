import torch

class MnistDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file: str, transform=None):
        self.data = []
        self.transform = transform

        self.__load_mnist_csv(csv_file)
        self.__transform()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __digit_to_one_hot_encoding(self, digit: int):
        encoding = [0] * 10
        encoding[digit] = 1
        return encoding

    def __load_mnist_csv(self, path: str):
        self.data.clear()
        with open(path, 'r') as f:
            for line in f:
                nums = [int(x) for x in line.split(',')]
                self.data.append({'label_str': str(nums[0]),
                                  'label': self.__digit_to_one_hot_encoding(nums[0]),
                                  'pixels': nums[1:]})

    def __transform(self):
        if self.transform is not None:
            for i in range(self.__len__()):
                self.data[i] = self.transform(self.data[i])
