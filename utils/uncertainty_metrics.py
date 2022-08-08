import numpy as np
import pickle
import pandas as pd
import matplotlib.patches as mpl_patches
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, auc
sns.set_style("whitegrid")

def histogram_intersection(data1, data2, nbins=100):
    h1 = np.histogram(data1, density=True, bins=[i * 0.01 for i in range(0, nbins + 1)])[0]
    h2 = np.histogram(data2, density=True, bins=[i * 0.01 for i in range(0, nbins + 1)])[0]

    return np.sum(np.minimum(h1, h2)) / 100


def uncertainty_box_plot(y_true, y_pred, **metrics):

    # correct/incorrect color palette ('green', 'red', 'green', 'red' ...)
    sns.set_palette(sns.color_palette("prism"))

    # create a dataframe which the desired data(metrics, y_true, y_pred, status - correct/incorrect)
    df_data = metrics.copy()
    for key, value in {"y_true": y_true,
                       "y_pred": y_pred,
                       "status": np.where((y_true == y_pred) == True, "correct", "incorrect")}.items():
        df_data[key] = value
    df = pd.DataFrame(df_data)

    fig, axes = plt.subplots(2, len(metrics), figsize=(12, 8))
    for idx, (key, value) in enumerate(metrics.items()):
        subfig = sns.boxplot(data=df,
                             y=key,
                             x="status",
                             showfliers=False,
                             width=0.35,
                             ax=axes[0, idx])
        subfig.set(xlabel=None)
        subfig.set(ylabel=None)
        subfig.set(xticklabels=[])
        subfig.set_title(key.replace("_", " ").capitalize())

    for idx, (key, value) in enumerate(metrics.items()):
        subfig = sns.histplot(np.ma.array(df[key], mask=np.invert(y_true == y_pred)).compressed(), stat="probability",
                              kde=True, color=sns.color_palette("prism")[0], line_kws={'linewidth': 3}, bins=100,
                              ax=axes[1, idx])
        subfig = sns.histplot(np.ma.array(df[key], mask=(y_true == y_pred)).compressed(), stat="probability", kde=True,
                              color=sns.color_palette("prism")[1], line_kws={'linewidth': 3}, bins=100, ax=axes[1, idx])

        # subfig = sns.histplot(data=df, x=key, hue="status", stat="probability", kde=True, line_kws={'linewidth': 3} , bins=100, ax=axes[1, idx])
        # subfig = sns.histplot(data=df_2, x=key, hue="status", stat="probability", kde=True, bins=[i * 0.01 for i in range(0, 101)], ax=axes[1, idx])

        hist_intersection = histogram_intersection(np.ma.array(df[key], mask=np.invert(y_true == y_pred)).compressed(),
                                                   np.ma.array(df[key], mask=(y_true == y_pred)).compressed())

        subfig.legend([mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)],
                      [f"Hist intersection: {hist_intersection:.2}"],
                      loc="best", fontsize=12, fancybox=True, shadow=True, handlelength=0, handletextpad=0)

        subfig.set(xlabel=None)
        subfig.set(ylabel=None)

    custom_lines = [Patch(facecolor=sns.color_palette("prism")[0]),
                    Patch(facecolor=sns.color_palette("prism")[1])]
    axes[0, 0].legend(custom_lines, ["correct", "incorrect"], fontsize=12, fancybox=True, shadow=True)
    axes[0, 0].set_ylabel("Uncertainty values", size="large")
    axes[1, 0].set_ylabel("Normalized KDE/Probability histogram", size="large")
    fig.tight_layout()

    return fig


def uncertainty_curve(y_true, y_pred, **metrics):
    sns.set_palette(sns.color_palette("Set1"))

    for name, metric in metrics.items():
        accuracy = []

        ideal_curve = [(len((y_true == y_pred)[(y_true == y_pred)]) + idx) / len((y_true == y_pred)) if (len(
            (y_true == y_pred)[(y_true == y_pred)]) + idx) / len((y_true == y_pred)) < 1 else 1 for idx in
                       range(len(y_true))]

        plt.plot((np.array(range(len(y_true))) * 100) / len(y_true),
                 ideal_curve,
                 linewidth=2,
                 color='black',
                 linestyle="--",
                 label="Perfect ordering"
                 )
        # sort predictions by uncertainty
        metric, y_true_ord, y_pred_ord = (list(t) for t in zip(*sorted(zip(metric, y_true, y_pred), reverse=True)))

        for idx in range(len(y_true_ord)):
            accuracy.append(accuracy_score(y_true_ord, y_pred_ord))
            y_pred_ord[idx] = y_true_ord[idx]

        au = auc(np.array(range(len(accuracy))) / len(accuracy), accuracy)
        nau = au / auc(np.array(range(len(ideal_curve))) / len(ideal_curve), ideal_curve)

        plt.plot((np.array(range(len(accuracy))) * 100) / len(accuracy),
                 accuracy,
                 linewidth=2,
                 label=f"{name.replace('_', ' ').capitalize()} - AUC: {au:.4f}\n"
                       f"{' '.rjust(len(name), ' ')} - NAUC: {nau:.4f}")

    plt.xlabel("Percentage of asked samples (%)")
    plt.ylabel("Accuracy")
    plt.legend(fontsize=12, fancybox=True, shadow=True)

    return plt.figure(1)