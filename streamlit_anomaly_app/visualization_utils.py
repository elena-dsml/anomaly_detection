from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np


def plot_clusters(X, cluster_labels, title="GMM Clusters"):
    fig = plt.figure(figsize=(7, 4))
    palette = sns.color_palette("Set2", len(np.unique(cluster_labels)))
    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=cluster_labels, palette=palette, s=60, edgecolor='k')

    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Cluster", loc="best")
    plt.grid(True)
    plt.tight_layout()

    return fig


def plot_anomaly_scores(anomaly_scores, title="Isolation Forest Anomaly Scores", threshold=None):
    fig = plt.figure(figsize=(6, 3))
    sns.histplot(anomaly_scores, bins=50, kde=True, color='orangered')
    plt.title(title)
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")

    if threshold:
        plt.axvline(threshold, color='black', linestyle='--', label=f"Threshold = {threshold:.4f}")
        plt.legend()

    plt.grid(True)
    plt.tight_layout()

    return fig

def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=False, title="Confusion Matrix"):
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
        cmap = 'YlGnBu'
    else:
        fmt = 'd'
        cmap = 'Blues'

    fig, ax = plt.subplots(figsize=(5, 3))

    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, xticklabels=labels, yticklabels=labels, ax=ax)

    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    fig.tight_layout()

    return fig
