

import random
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

def stratified_group_k_fold(X, y, groups, k, seed=None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices













# # X_train_non_null.shape, X_test_non_null.shape, np.unique(X_train_non_null.columns).shape
# kf = GroupKFold(n_splits=n_splits)

# oof_train = np.zeros((X_train.shape[0]))
# oof_test = np.zeros((X_test.shape[0], n_splits))

# i = 0
# print("running for {} splits".format(n_splits))
# #     for train_idx, valid_idx in kf.split(X_train, X_train['AdoptionSpeed'].values):
# for train_idx, valid_idx in kf.split(X_train, X_train['AdoptionSpeed'].values, X_train.RescuerID):
#     print(train_idx[:10])
#     print(valid_idx[:10])
#     print("="*20)
# print("*"*20)
# print("*"*20)
# for train_idx, valid_idx in stratified_group_k_fold(X=X_train, y=X_train['AdoptionSpeed'].astype('int64').values, 
#                                                         groups= np.array(X_train.RescuerID.values), 
#                                                         k=5, seed=2019):
#     print(train_idx[:10])
#     print(valid_idx[:10])
#     print("="*20)