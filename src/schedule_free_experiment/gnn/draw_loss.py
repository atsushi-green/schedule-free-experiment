import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

"""csv example
Epoch,Train Loss,Val Loss
1,1.9455,1.9456
2,1.9458,1.9456
3,1.9454,1.9456
4,1.9453,1.9456
5,1.9455,1.9438
6,1.9422,1.9427
7,1.9397,1.9395
8,1.9341,1.9359
9,1.9288,1.9309
10,1.9208,1.9255
11,1.9104,1.9189
12,1.8993,1.9115
13,1.8790,1.9032
"""
BLUE = "#1E90FF"
RED = "#F26649"

LINE_COLOR = {"scheduler": BLUE, "scheduleFree": RED}
LINE_STYLE = {"Train": "solid", "Val": "dashed"}


def plot_loss_comparison(base_dir: Path) -> None:
    csv_files = list(base_dir.glob("*.csv"))
    condition_set = set()
    datasetname_set = set()
    lr_set = set()

    for csv_file in csv_files:
        # scheduleFree_loss_history_Cora_init_lr=0.1.csv のようなファイル名を読み込む
        filename = csv_file.name
        condition = filename.split("_")[0]  # "scheduleFree"か"scheduler"か
        datasetname = filename.split("_")[3]  # "Cora"か"Planetoid"か"Pubmed"か
        # ".csv"の4文字を除いて取得。学習率の初期値
        lr = float(filename.split("=")[1][:-4])

        condition_set.add(condition)
        datasetname_set.add(datasetname)
        lr_set.add(lr)

    conditions = list(sorted(condition_set))
    datasetnames = list(sorted(datasetname_set))
    learning_rates = list(sorted(lr_set))

    for condition in conditions:
        for datasetname in datasetnames:
            fig, ax = plt.subplots()
            for lr in learning_rates:
                file_path = (
                    base_dir
                    / f"{condition}_loss_history_{datasetname}_init_lr={lr}.csv"
                )
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    # ax.plot(
                    #     df["Epoch"],
                    #     df["Train Loss"],
                    #     label=f"{condition}_{datasetname}_{lr}_Train",
                    # )
                    ax.plot(
                        df["Epoch"],
                        df["Val Loss"],
                        label=f"{condition}_{datasetname}_{lr}_Val",
                    )

            plt.title(f"Loss Comparison for {condition}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            ax.set_ylim(0, 2)
            plt.savefig(base_dir / f"{condition}_{datasetname}_loss_comparison.png")
            plt.close(fig)

    # 同じ学習率同士で比較する作図
    for datasetname in datasetnames:
        for lr in learning_rates:
            fig, ax = plt.subplots()
            for condition in conditions:
                file_path = (
                    base_dir
                    / f"{condition}_loss_history_{datasetname}_init_lr={lr}.csv"
                )
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    ax.plot(
                        df["Epoch"],
                        df["Train Loss"],
                        label=f"{condition}_{datasetname}_{lr}_Train",
                        color=LINE_COLOR[condition],
                        linestyle=LINE_STYLE["Train"],
                    )
                    ax.plot(
                        df["Epoch"],
                        df["Val Loss"],
                        label=f"{condition}_{datasetname}_{lr}_Val",
                        color=LINE_COLOR[condition],
                        linestyle=LINE_STYLE["Val"],
                    )

            plt.title(f"Loss Comparison for {condition} lr={lr}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            ax.set_ylim(0, 2)
            plt.savefig(base_dir / f"{datasetname}_lr={lr}_loss_comparison.png")
            plt.close(fig)


base_dir = Path("gnn_loss_history")
plot_loss_comparison(base_dir)
