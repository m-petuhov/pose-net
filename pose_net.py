import torch.nn as nn
import argparse
import os, ssl
import pandas as pd
import torchvision
import numpy as np

from torchvision.models import googlenet
from common_functions import *
from dataset import RelocalizationDataset


class PoseNet(nn.Module):
    __out_classes__ = 7

    def __init__(self):
        super(PoseNet, self).__init__()

        self.base_model = googlenet(pretrained=True, aux_logits=True)
        self.base_model.aux1.fc2 = nn.Linear(1024, self.__out_classes__)
        self.base_model.aux2.fc2 = nn.Linear(1024, self.__out_classes__)
        self.base_model.fc = nn.Linear(1024, self.__out_classes__)

    def forward(self, x):
        return self.base_model._forward(x)


if __name__ == '__main__':
    if not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Read args
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-weights', type=str, required=True)
    parser.add_argument('--path-test-dir', type=str, required=True)
    args = parser.parse_args()

    # Prepare model
    model = PoseNet().to(DEVICE)

    load_weights(model, args.path_weights, DEVICE)

    # Upload data
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.RandomCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # TODO global mean and std

    info = pd.read_csv(args.path_test_dir + '/info.csv')
    dataset = RelocalizationDataset(info.values, args.path_test_dir + '/images', transform=transform, mode='test')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)

    # Predict
    predictions = predict(model, data_loader, DEVICE)
    labels = torch.Tensor([np.array(info.values[i][3:10], dtype=np.float16) for i in range(2)])

    print("Predictions:", predictions)
    print("Expected:", labels)

