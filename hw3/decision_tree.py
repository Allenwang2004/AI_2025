import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from typing import List, Tuple

"""
Notice:
    1) You can't add any additional package
    2) You can add or remove any function "except" fit, _build_tree, predict
    3) You can ignore the suggested data type if you want
"""

class ConvNet(nn.Module): # Don't change this part!
    def __init__(self):
        super(ConvNet, self).__init__()
        self.model = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=300)

    def forward(self, x):
        x = self.model(x)
        return x
    
class DecisionTree:
    def __init__(self, max_depth=16, min_samples_split=10, min_samples_leaf=5, min_impurity_decrease=1e-7):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.data_size = X.shape[0]
        total_steps = 2 ** self.max_depth
        self.progress = tqdm(total=total_steps, desc="Growing tree", position=0, leave=True)
        self.tree = self._build_tree(X, y, depth=0)
        self.progress.close()

    def _build_tree(self, X: pd.DataFrame, y: np.ndarray, depth: int):
        if (len(np.unique(y)) == 1 or 
            depth >= self.max_depth or 
            len(X) < self.min_samples_split):
            return np.bincount(y).argmax()

        best_feature_index, best_threshold, best_gain = self._best_split(X, y)

        if best_feature_index is None or best_gain < self.min_impurity_decrease:
            return np.bincount(y).argmax()

        left_X, left_y, right_X, right_y = self._split_data(X, y, best_feature_index, best_threshold)

        if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
            return np.bincount(y).argmax()

        self.progress.update(1)

        return {
            'feature_index': best_feature_index,
            'threshold': best_threshold,
            'left': self._build_tree(left_X, left_y, depth + 1),
            'right': self._build_tree(right_X, right_y, depth + 1)
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = [self._predict_tree(row, self.tree) for _, row in X.iterrows()]
        return np.array(predictions)

    def _predict_tree(self, x, tree_node):
        if not isinstance(tree_node, dict):
            return tree_node

        feature_val = x[tree_node['feature_index']]

        if feature_val < tree_node['threshold']:
            return self._predict_tree(x, tree_node['left'])
        else:
            return self._predict_tree(x, tree_node['right'])

    def _split_data(self, X: pd.DataFrame, y: np.ndarray, feature_index: int, threshold: float):
        left_mask = X.iloc[:, feature_index] < threshold
        right_mask = ~left_mask
        left_dataset_X = X[left_mask]
        left_dataset_y = y[left_mask]
        right_dataset_X = X[right_mask]
        right_dataset_y = y[right_mask]
        return left_dataset_X, left_dataset_y, right_dataset_X, right_dataset_y

    def _best_split(self, X: pd.DataFrame, y: np.ndarray):
        best_gain = -1
        best_feature_index, best_threshold = None, None

        for feature_index in range(X.shape[1]):
            threshold = X.iloc[:, feature_index].median()

            left_mask = X.iloc[:, feature_index] < threshold
            right_mask = ~left_mask

            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue

            gain = self._information_gain(y, y[left_mask], y[right_mask])

            if gain > best_gain:
                best_gain = gain
                best_feature_index = feature_index
                best_threshold = threshold

        return best_feature_index, best_threshold, best_gain

    def _entropy(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _information_gain(self, parent_y, left_y, right_y):
        n = len(parent_y)
        n_left = len(left_y)
        n_right = len(right_y)

        if n == 0:
            return 0

        entropy_parent = self._entropy(parent_y)
        entropy_left = self._entropy(left_y)
        entropy_right = self._entropy(right_y)

        child_entropy = (n_left / n) * entropy_left + (n_right / n) * entropy_right
        return entropy_parent - child_entropy
    
    def prune(self, X_val: pd.DataFrame, y_val: np.ndarray):
        print("🧹 Start pruning...")
        self._prune_node(self.tree, X_val, y_val)

    def _prune_node(self, node, X_val: pd.DataFrame, y_val: np.ndarray):
        if not isinstance(node, dict):
            return node

        node['left'] = self._prune_node(node['left'], X_val, y_val)
        node['right'] = self._prune_node(node['right'], X_val, y_val)

        if not isinstance(node['left'], dict) and not isinstance(node['right'], dict):
            preds_before = self._predict_batch(X_val, self.tree)

            majority_label = np.bincount(y_val).argmax()
            backup_node = node.copy()

            pruned_tree = majority_label

            tmp = self.tree
            self.tree = pruned_tree
            preds_after = np.full_like(y_val, majority_label)

            acc_before = (preds_before == y_val).mean()
            acc_after = (preds_after == y_val).mean()

            if acc_after >= acc_before:
                print(f"✅ Pruned node with acc_before={acc_before:.4f}, acc_after={acc_after:.4f}")
                return pruned_tree
            else:
                self.tree = tmp
                return backup_node

        return node

def _predict_batch(self, X: pd.DataFrame, tree):
    """
    Use a given tree to predict a batch of data
    """
    preds = []
    for _, row in X.iterrows():
        preds.append(self._predict_tree(row, tree))
    return np.array(preds)

def get_features_and_labels(model: ConvNet, dataloader: DataLoader, device) -> Tuple[List, List]:
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, batch_labels in tqdm(dataloader, desc='Extracting features'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
            labels.append(batch_labels.numpy())
    features = np.vstack(features)
    labels = np.concatenate(labels)
    return pd.DataFrame(features), labels

def get_features_and_paths(model: ConvNet, dataloader: DataLoader, device) -> Tuple[List, List]:
    model.eval()
    features = []
    paths = []
    with torch.no_grad():
        for inputs, batch_paths in tqdm(dataloader, desc='Extracting test features'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
            paths.extend(batch_paths)
    features = np.vstack(features)
    return pd.DataFrame(features), paths
