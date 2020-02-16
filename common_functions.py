import torch
import copy

from time import gmtime, strftime


def load_weights(model, path, device):
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    model.eval()


def save_weights(model, path, val_acc):
    name = strftime("%Yy.%mm.%dd.%Hh.%Mm", gmtime())
    model_weights = copy.deepcopy(model.state_dict())
    torch.save(model_weights, f"{path}/{round(val_acc, 3)}___{name}.pth")


def predict(model, data_loader, device):
    model.eval()
    res = []

    with torch.set_grad_enabled(False):
        for batch in data_loader:
            predictions = sum(model(batch[:, i].to(device))[0] for i in range(3)) / 3
            res.append(predictions)

    return res
