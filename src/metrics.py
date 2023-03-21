import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

def compute_metrics(y_true, y_pred, cls_names):
    f1 = compute_f1(y_true, y_pred)
    acc = compute_accuracy(y_true, y_pred)
    cm = compute_confusion_matrix(y_true, y_pred, cls_names)

    return f1, acc, cm

def compute_f1(y_true, y_pred, average='macro') -> float:
    return f1_score(y_true, y_pred, average=average)

def compute_accuracy(y_true, y_pred) -> float:
    return np.mean(y_true == y_pred)

def compute_confusion_matrix(y_true, y_pred, cls_names) -> plt.Figure:
    conf_mat = confusion_matrix(y_true, y_pred)     # Compute CM
    conf_mat_norm = conf_mat / conf_mat.sum(axis=1, keepdims=True)   # Normalize CM

    cls_names = [c.replace('_', '\n').replace(' ', '\n') for c in cls_names]   # Clean names

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(conf_mat_norm,
                annot=True,
                fmt='.2%',
                cmap=plt.get_cmap('Greys'),
                annot_kws={"size": 10},
                yticklabels=cls_names,
                xticklabels=cls_names,
                ax=ax)

    # title = f"Confusion Matrix"
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.tick_params(labelsize=10, length=0)
    # ax.set_title(title, size=18, pad=20)
    ax.set_xlabel('Predicted Values', size=14)
    ax.set_ylabel('GT Values', size=14)

    samples = conf_mat.flatten().tolist()
    samples = [str(s) for s in samples]
    # samples = ['' for s in samples if s=='0']
    # samples = samples.replace('0', '')

    for text_elt, additional_text in zip(ax.texts, samples):
        ax.text(*text_elt.get_position(), '\n' + additional_text, color=text_elt.get_color(),
                ha='center', va='top', size=10)

    return fig

