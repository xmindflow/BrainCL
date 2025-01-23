import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json  # For loading the data from the JSON file
from glob import glob  # For finding all JSON files in a directory
import os
import numpy as np


def calculate_metrics(array_2d, noOfEpisodes):
    N = noOfEpisodes
    count_trangle = 0.5 * N * (N - 1)
    count_training_with_diag = 0.5 * N * (N + 1)

    row_averages = np.round(np.mean(array_2d, axis=1), 4)
    overall_avg_acc = np.average(array_2d)
    avg_acc_last_episode = np.average(array_2d[-1])
    lower_triangular = np.tril(array_2d)
    avg_lower_triangular_with_diag = np.sum(lower_triangular) / count_training_with_diag

    temp = []
    for i in range(noOfEpisodes - 1):
        diag = array_2d[i, i]
        candidate = array_2d[i + 1 :, i]
        temp.extend(candidate - diag)
    bwt_rodriguez_gonzalez = np.average(temp)

    upper_triangular = np.triu(array_2d, k=1)
    fwt_rodriguez = np.sum(upper_triangular) / count_trangle

    return {
        "row_averages": row_averages,
        "overall_avg_acc": overall_avg_acc,
        "avg_acc_last_episode": avg_acc_last_episode,
        "avg_lower_triangular_with_diag": avg_lower_triangular_with_diag,
        "bwt_rodriguez_gonzalez": bwt_rodriguez_gonzalez,
        "fwt_rodriguez": fwt_rodriguez,
    }


def print_some_CL_metrices(accuracy_values, noOfEpisodes):
    array_2d = np.array(accuracy_values).reshape(noOfEpisodes, noOfEpisodes)
    print("array_2d=", array_2d)
    metrics = calculate_metrics(array_2d, noOfEpisodes)

    print("Row-wise average values=", metrics["row_averages"])
    print("overall avg acc ={:.2f}".format(metrics["overall_avg_acc"]))
    print("avg acc at last episode={:.2f}".format(metrics["avg_acc_last_episode"]))
    print(
        "Average of lower triangular with diag={:.2f}".format(
            metrics["avg_lower_triangular_with_diag"]
        )
    )
    print("BWT Rodriguez/Gonzalez ={:.2f}".format(metrics["bwt_rodriguez_gonzalez"]))
    print("FWT Rodriguez={:.2f}".format(metrics["fwt_rodriguez"]))

    return metrics


def plot_heatmap(parent_dir, save_image_path, sequence=0):
    title_name = os.path.basename(parent_dir)
    print("title_name=", title_name)
    json_files = sorted(glob(os.path.join(parent_dir, "*.json")))
    if sequence == 0:
        index = ["BRATS", "ATLAS", "MSSEG", "ISLES", "WMH"]
    elif sequence == 1:
        index = ["MSSEG", "BRATS", "ISLES", "WMH", "ATLAS"]
    else:
        raise ValueError("Invalid sequence value. It should be either 0 or 1")
    all_modalities = ["FLAIR", "T1", "T1c", "T2", "PD", "DWI"]
    modalities = {
        "BRATS": ["FLAIR", "T1", "T1c", "T2"],
        "ATLAS": ["T1"],
        "MSSEG": ["FLAIR", "T1", "T1c", "T2", "PD"],
        "ISLES": ["FLAIR", "T1", "T2", "DWI"],
        "WMH": ["FLAIR", "T1"],
    }

    data = {}
    for i, json_file in enumerate(json_files):
        with open(json_file, "r") as f:
            results = json.load(f)
        temp_results = [results[name] for name in index]
        data[index[i]] = temp_results

    df = pd.DataFrame(data, index=index).T

    formatted_modalities = {
        dataset: ", ".join(
            [modality if modality in mods else "-" for modality in all_modalities]
        )
        for dataset, mods in modalities.items()
    }

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(24, 12), gridspec_kw={"width_ratios": [5, 2]}
    )
    ax1: plt.Axes

    sns_plot = sns.heatmap(
        df,
        annot=True,
        cmap="viridis",
        linewidths=0.5,
        fmt=".4f",
        cbar=True,
        annot_kws={"size": 18},
        ax=ax1,
    )
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position("top")
    ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=12)
    ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=12)

    for y, mods in enumerate(formatted_modalities.values()):
        ax1.text(
            -0.3,
            y + 0.5,
            mods,
            va="center",
            ha="right",
            fontsize=14,
            color="black",
            clip_on=False,
        )

    fig.suptitle(title_name, fontsize=20, y=0.98)
    ax1.set_xlabel("Tested On")
    ax1.set_ylabel("Trained On")

    metrics = print_some_CL_metrices(df.values, 5)
    metrics_text = "\n".join(
        [
            f"Row-wise average values: {metrics['row_averages']*100}",
            f"Overall avg acc: {metrics['overall_avg_acc']*100:.2f}",
            f"Avg acc at last episode: {metrics['avg_acc_last_episode']*100:.2f}",
            f"Avg lower triangular with diag: {metrics['avg_lower_triangular_with_diag']*100:.2f}",
            f"BWT Rodriguez/Gonzalez: {metrics['bwt_rodriguez_gonzalez']*100:.2f}",
            f"FWT Rodriguez: {metrics['fwt_rodriguez']*100:.2f}",
        ]
    )

    ax2.axis("off")
    ax2.text(0, 0.5, metrics_text, ha="left", va="center", fontsize=16, wrap=True)

    # plt.tight_layout()
    fig.tight_layout(rect=[0.03, 0, 1, 0.95])
    plt.savefig(os.path.join(save_image_path, f"{title_name}.png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    
    parent_dir = "/home/say26747/Alex/CL/avalanche/Sequence1/cumulative/cumulative_optim_adam_lr_0.001_bs_4_epochs_400_drop_1" # Path to the directory containing the JSON files
    plot_heatmap(parent_dir, save_image_path=parent_dir, sequence=1)
