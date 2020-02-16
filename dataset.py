import numpy as np
import torch
import os
import torchvision as torchvision

from torch.utils.data import Dataset

DATA_MODES = ['train', 'test']


class RelocalizationDataset(Dataset):

    def __init__(self, info, path_data, transform=None,
                 loader=torchvision.datasets.folder.default_loader,
                 mode='train'):
        super().__init__()

        self.info = info
        self.path_data = path_data
        self.transform = transform
        self.loader = loader
        self._len = info.shape[0]
        self.mode = mode

        if self.mode not in DATA_MODES:
            print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
            raise NameError

    def __getitem__(self, index):
        images = []

        for name in [-3, -2, -1]:
            images.append(self.loader(os.path.join(self.path_data, self.info[index][name])))

            if self.transform is not None:
                images[-1] = self.transform(images[-1])

        if self.mode == 'train':
            return (torch.Tensor([np.array(image) for image in images]),
                    torch.Tensor(np.array(self.info[index][3:10], dtype=np.float16)))
        else:
            return torch.Tensor([np.array(image) for image in images])

    def __len__(self):
        return self._len
