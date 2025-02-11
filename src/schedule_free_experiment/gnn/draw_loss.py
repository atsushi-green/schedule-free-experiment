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
LINE_STYLE = {
    "Train": "solid",
    "Val": "dashed",
    "scheduler": "solid",
    "scheduleFree": "dashed",
}
# 図のy軸の最大値（誤差の大きさをどこまで見るか）
YMAX_DICT = {"Pubmed": 2, "Planetoid": 2, "Cora": 2.5}


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

    n_colors = len(learning_rates)
    cmap = plt.get_cmap("jet", n_colors)
    color_list = [cmap(i) for i in range(n_colors)]

    # 全部描画
    for datasetname in datasetnames:
        ymax = YMAX_DICT[datasetname]
        fig, ax = plt.subplots(figsize=(10, 5))
        fig_acc, ax_acc = plt.subplots(figsize=(10, 5))
        for condition in conditions:
            for i, lr in enumerate(learning_rates):
                file_path = (
                    base_dir
                    / f"{condition}_loss_history_{datasetname}_init_lr={lr}.csv"
                )
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)

                    ax.plot(
                        df["Epoch"],
                        df["Test Loss"],
                        label=f"{condition}_{lr}_Test",
                        # color=LINE_COLOR[condition],
                        color=color_list[i],
                        linestyle=LINE_STYLE[condition],
                        linewidth=0.8,
                        # alpha=alpha_list[i],
                        # linewidth=width_list[i],
                    )
                    ax_acc.plot(
                        df["Epoch"],
                        df["Test Acc"],
                        label=f"{condition}_{lr}_Test",
                        # color=LINE_COLOR[condition],
                        color=color_list[i],
                        linestyle=LINE_STYLE[condition],
                        linewidth=0.8,
                        # alpha=alpha_list[i],
                        # linewidth=width_list[i],
                    )
        # 命名を誤ったのでここで修正
        if datasetname == "Planetoid":
            datasetname = "CiteSeer"
        ax.set_title(f"Loss Comparison for {datasetname} Dataset")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test Loss")
        ax.legend(bbox_to_anchor=(1.05, 0.5), borderaxespad=0, loc="center left")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, ymax)
        fig.tight_layout()
        fig.savefig(base_dir / f"{datasetname}_loss_comparison.png")
        plt.close(fig)

        ax_acc.set_title(f"Accuracy Comparison for {datasetname} Dataset")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Test Accuracy")
        ax_acc.legend(bbox_to_anchor=(1.05, 0.5), borderaxespad=0, loc="center left")
        ax_acc.set_xlim(0, 100)
        ax_acc.set_ylim(0, 1)
        fig_acc.tight_layout()
        fig_acc.savefig(base_dir / f"{datasetname}_accuracy_comparison.png")
        plt.close(fig_acc)

    # conditionごとに比較する作図
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

                    ax.plot(
                        df["Epoch"],
                        df["Test Loss"],
                        label=f"{condition}_{datasetname}_{lr}_Test",
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
