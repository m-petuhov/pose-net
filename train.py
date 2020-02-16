import argparse
import pandas as pd
import torch
import torchvision
from tqdm import tqdm

from common_functions import save_weights, load_weights
from dataset import RelocalizationDataset
from sklearn.model_selection import train_test_split

from pose_net import PoseNet


def loss(y, y_pred):
    return torch.norm(y[:3] - y_pred[:3]) + 157 * torch.norm(y[3:] - y_pred[3:] / torch.norm(y_pred[3:] ))


def fit_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train(True)

    running_loss = 0.0
    processed_data = 0

    for batch, labels in data_loader:
        batch = batch.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        prediction_x, prediction_aux1, prediction_aux2 = [sum(model(batch[:, i].to(device))[j] for i in range(3)) / 3
                                                          for j in range(3)]
        loss_x = loss_fn(prediction_x, labels)
        loss_aux1 = loss_fn(prediction_aux1, labels)
        loss_aux2 = loss_fn(prediction_aux2, labels)

        loss_x.backward()
        loss_aux1.backward()
        loss_aux2.backward()
        optimizer.step()

        running_loss += loss_x.item() * batch.size(0)
        processed_data += batch.size(0)

    train_loss = running_loss / processed_data

    return train_loss


def eval_epoch(model, data_loader, loss_fn, device):
    model.eval()

    running_loss = 0.0
    processed_size = 0

    for batch, labels in data_loader:
        batch = batch.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            prediction_x = sum(model(batch[:, i].to(device))[0] for i in range(3)) / 3
            loss_x = loss_fn(prediction_x, labels)

        running_loss += loss_x.item() * batch.size(0)
        processed_size += batch.size(0)

    loss = running_loss / processed_size

    return loss


def train(data_loader, model, epochs, device, save_path=None):
    history = []
    log_template = "\nLog epoch {ep:03d}:\n\ttrain_loss: {t_loss:0.4f} \n\tval_loss {v_loss:0.4f}"

    with tqdm(desc="epoch", total=epochs) as pbar_outer:
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = loss

        for epoch in range(epochs):
            train_loss = fit_epoch(model, data_loader['train'],
                                   loss_fn, optimizer, device)
            val_loss = eval_epoch(model, data_loader['val'], loss_fn, device)

            history.append((train_loss, val_loss))
            tqdm.write(log_template.format(ep=epoch + 1, t_loss=train_loss,
                                           v_loss=val_loss))
            pbar_outer.update(1)

            if save_path is not None:
                save_weights(model, save_path, val_loss)

    return history


if __name__ == '__main__':
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Read args
    parser = argparse.ArgumentParser()
    parser.add_argument('--module-name', type=str, required=True)
    parser.add_argument('--path-weights', type=str, required=False)
    parser.add_argument('--path-train-dir', type=str, required=True)
    args = parser.parse_args()

    # Upload data
    info = pd.read_csv(args.path_train_dir + '/info.csv')

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.RandomCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    X_train, X_test = train_test_split(info.values, test_size=0.2, random_state=123)

    image_datasets = {
        'train': RelocalizationDataset(
            X_train, args.path_train_dir + '/images', transform=transform),
        'val': RelocalizationDataset(
            X_test, args.path_train_dir + '/images', transform=transform)
    }

    data_loaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=10,
                                       shuffle=True)
        for x in ['train', 'val']
    }

    if args.module_name == 'pose-net':
        model = PoseNet()
    else:
        raise NameError

    if args.path_weights is not None:
        load_weights(model, args.path_weights, DEVICE)

    history = train(data_loaders, model=model, epochs=1, device=DEVICE, save_path='weights')
