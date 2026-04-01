import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from collections import Counter
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.neighbors import KNeighborsClassifier


def save_evaluation_to_excel(subject_id, model_name, 
                              results_train, results_test, results_heldout, 
                              clustering_train, clustering_test, clustering_heldout, 
                              chance_train, chance_test, chance_heldout,
                              save_path="results/evaluation_summary.xlsx"):
    """
    Save all evaluation results to an Excel file.

    Args:
        subject_id (str): Subject/session identifier
        model_name (str): Name of the saved model
        results_* (dict): Output of evaluate_knn() for train/test/held-out
        clustering_* (dict): Clustering metrics for train/test/held-out
        chance_* (dict): Uniform + majority class chance levels for train/test/held-out
        save_path (str): Path to output Excel file
    """
    row = {
        "Subject": subject_id,
        "Model Name": model_name,

        # Train
        "Train Accuracy": results_train["accuracy"],
        "Train Chance (Uniform)": chance_train["uniform_chance"],
        "Train Chance (Majority)": chance_train["majority_class_chance"],
        "Silhouette (Train)": clustering_train["silhouette"],
        "DB Index (Train)": clustering_train["davies_bouldin"],
        "CH Score (Train)": clustering_train["calinski_harabasz"],

        # Test
        "Test Accuracy": results_test["accuracy"],
        "Test Chance (Uniform)": chance_test["uniform_chance"],
        "Test Chance (Majority)": chance_test["majority_class_chance"],
        "Silhouette (Test)": clustering_test["silhouette"],
        "DB Index (Test)": clustering_test["davies_bouldin"],
        "CH Score (Test)": clustering_test["calinski_harabasz"],

        # Held-out
        "Held-Out Accuracy": results_heldout["accuracy"],
        "Held-Out Chance (Uniform)": chance_heldout["uniform_chance"],
        "Held-Out Chance (Majority)": chance_heldout["majority_class_chance"],
        "Silhouette (Held-Out)": clustering_heldout["silhouette"],
        "DB Index (Held-Out)": clustering_heldout["davies_bouldin"],
        "CH Score (Held-Out)": clustering_heldout["calinski_harabasz"]
    }

    df_new = pd.DataFrame([row])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        df_existing = pd.read_excel(save_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_excel(save_path, index=False)
    print(f"✅ Evaluation saved to {save_path}")


def evaluate_clustering_metrics(embeddings, labels):
    """
    Compute clustering quality metrics.

    Args:
        embeddings (np.ndarray): (N, D) array of latent vectors
        labels (list or array): Region labels for each embedding

    Returns:
        dict: Metric name → score
    """
    return {
        "silhouette": silhouette_score(embeddings, labels),
        "davies_bouldin": davies_bouldin_score(embeddings, labels),
        "calinski_harabasz": calinski_harabasz_score(embeddings, labels)
    }

def plot_and_save_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """
    Plots and saves a labeled confusion matrix.

    Args:
        y_true (list or array): Ground truth labels
        y_pred (list or array): Predicted labels
        class_names (list): Unique class names in consistent order
        title (str): Title for the plot
        save_path (str): Path to save the image (e.g. 'results/cm_test.png')
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # Normalize by row

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 14})
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("True", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Confusion matrix saved to: {save_path}")


def compute_chance_levels(true_labels):
    num_classes = len(set(true_labels))
    uniform = 1 / num_classes

    class_counts = Counter(true_labels)
    majority_class_ratio = class_counts.most_common(1)[0][1] / len(true_labels)

    return {
        "uniform_chance": uniform,
        "majority_class_chance": majority_class_ratio
    }


def embed_segments_dict(segments_dict, model, device):
    """
    Extract embeddings and labels from a dictionary of labeled segments.

    Args:
        segments_dict (dict): {label: np.ndarray of shape (N_segments, T)}
        model (nn.Module): Trained model with .embed()
        device (torch.device): 'cuda' or 'cpu'

    Returns:
        embeddings (np.ndarray): (N, D) matrix of embeddings
        labels (list): Region label for each embedding
    """
    all_embeddings = []
    all_labels = []

    for label, segments in segments_dict.items():
        emb_list = _extract_embeddings(model, segments, label=label, device=device)
        for emb, region in emb_list:
            all_embeddings.append(emb)
            all_labels.append(region)

    return np.array(all_embeddings), all_labels


def evaluate_knn(train_embeddings, train_labels, test_embeddings, test_labels, k=5):
    """
    Perform KNN classification and return accuracy and confusion matrix.

    Args:
        train_embeddings (np.ndarray): (N_train, D) latent vectors
        train_labels (np.ndarray): (N_train,) region labels
        test_embeddings (np.ndarray): (N_test, D) latent vectors
        test_labels (np.ndarray): (N_test,) region labels
        k (int): Number of neighbors for KNN

    Returns:
        dict: accuracy, predictions, confusion matrix, classification report
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_embeddings, train_labels)

    preds = knn.predict(test_embeddings)
    acc = accuracy_score(test_labels, preds)
    cm = confusion_matrix(test_labels, preds)
    report = classification_report(test_labels, preds, output_dict=True)

    return {
        "accuracy": acc,
        "predictions": preds,
        "confusion_matrix": cm,
        "report": report
    }


def _extract_embeddings(model, data_segments, label, device):
    """
    Pass a list of time-series segments through the encoder to extract embeddings.

    Args:
        model (nn.Module): Model with an `embed()` method returning embeddings.
        data_segments (list of arrays): 1D segments to embed.
        label (str or int): Label to associate with all embeddings.
        device (torch.device): Device to run inference on.

    Returns:
        list of (embedding, label): Each embedding is a 1D NumPy array.
    """
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for segment in data_segments:
            x = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, T)
            embedding = model.embed(x).cpu().numpy().squeeze()
            all_embeddings.append((embedding, label))
    return all_embeddings