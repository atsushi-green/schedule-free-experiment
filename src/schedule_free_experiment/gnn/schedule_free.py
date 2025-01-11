import csv

import torch
import torch_geometric.data
from gnn import GCN, Dataset
from schedulefree import RAdamScheduleFree
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


def main():
    # データセットの読み込み
    dataset_planetoid = Planetoid(
        root="data/Planetoid", name="CiteSeer", transform=NormalizeFeatures()
    )
    dataset_pubmed = Planetoid(
        root="data/Pubmed", name="Pubmed", transform=NormalizeFeatures()
    )
    dataset_cora = Planetoid(
        root="data/Cora", name="Cora", transform=NormalizeFeatures()
    )

    # 各条件で実験（データセット×学習率の初期値）
    for dataset, name in zip(
        [dataset_planetoid, dataset_pubmed, dataset_cora],
        ["Planetoid", "Pubmed", "Cora"],
    ):
        for init_lr in [0.0001, 0.001, 0.002, 0.004, 0.008, 0.01, 0.05, 0.1]:
            exp_name = f"{name}_init_lr={init_lr}"
            experiment(dataset, init_lr, exp_name)


def experiment(dataset: Dataset, init_lr: float, exp_name: str):
    data = dataset[0]

    print()
    print(data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCN(dataset, hidden_channels=64, dropout=0.5).to(device)
    data = data.to(device)
    params = model.parameters()
    optimizer = RAdamScheduleFree(params, lr=init_lr, betas=(0.9, 0.999))

    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = best_test_acc = 0
    loss_history = []
    print("Epoch, Train, Val, Test")
    for epoch in range(1, 401):
        train_loss = train(
            model=model,
            data=data,
            optimizer=optimizer,
            criterion=criterion,
        )
        val_loss = valid(
            model=model, data=data, criterion=criterion, optimizer=optimizer
        )

        test_loss = test(
            model=model, data=data, criterion=criterion, optimizer=optimizer
        )
        train_acc, val_acc, test_acc = calc_acc(model=model, data=data)

        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     best_test_acc = test_acc

        loss_history.append(
            [train_loss, val_loss, test_loss, train_acc, val_acc, test_acc]
        )
        print(
            f"{epoch:04d},{train_acc:.4f},{val_acc:.4f},{test_acc:.4f}\t{train_loss:.4f},{val_loss:.4f},{test_loss:.4f}"
        )

    # lossの保存
    with open(f"scheduleFree_loss_history_{exp_name}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Epoch",
                "Train Loss",
                "Val Loss",
                "Test Loss",
                "Train Acc",
                "Val Acc",
                "Test Acc",
            ]
        )
        for epoch, (
            train_loss,
            val_loss,
            test_loss,
            train_acc,
            val_acc,
            test_acc,
        ) in enumerate(loss_history, 1):
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.4f}",
                    f"{val_loss:.4f}",
                    f"{test_loss:.4f}",
                    f"{train_acc:.4f}",
                    f"{val_acc:.4f}",
                    f"{test_acc:.4f}",
                ]
            )


def train(
    model: torch.nn.Module, data: torch_geometric.data.data.Data, optimizer, criterion
):
    model.train()
    optimizer.train()

    optimizer.zero_grad()
    # out = model(data.x, data.edge_index)
    batch_size = 64
    train_indices = data.train_mask.nonzero(as_tuple=True)[0]
    # train_indicesをシャッフル
    train_indices = train_indices[torch.randperm(train_indices.size(0))]
    train_loss_value = 0
    for i in range(0, len(train_indices), batch_size):
        batch_indices = train_indices[i : i + batch_size]
        out = model(data.x, data.edge_index)
        train_loss = criterion(out[batch_indices], data.y[batch_indices])
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss_value += train_loss.item()

    # train_loss = criterion(out[data.train_mask], data.y[data.train_mask])
    # train_loss.backward()
    # optimizer.step()

    return train_loss_value


def valid(
    model: torch.nn.Module,
    data: torch_geometric.data.data.Data,
    criterion: callable,
    optimizer: torch.optim.Optimizer,
):
    model.eval()
    optimizer.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
    return val_loss


def test(
    model: torch.nn.Module,
    data: torch_geometric.data.data.Data,
    criterion: callable,
    optimizer: torch.optim.Optimizer,
):
    model.eval()
    optimizer.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        test_loss = criterion(out[data.test_mask], data.y[data.test_mask])
    return test_loss


def calc_acc(model: torch.nn.Module, data: torch_geometric.data.data.Data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    accs = []
    for _, mask in data("train_mask", "val_mask", "test_mask"):
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


if __name__ == "__main__":
    main()
