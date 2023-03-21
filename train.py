from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from scipy.stats import pearsonr, spearmanr, kendalltau

from data import names


def cross_validate(dataset, model, names):
    data = pd.read_csv(dataset)
    videos = data.video.unique()

    stats = defaultdict(lambda: defaultdict(list))
    for fold, (train_idxs, test_idxs) in enumerate(KFold(n_splits=5).split(videos)):
        data_train = data[data.video.isin(videos[train_idxs])]
        model.fit(data_train.loc[:, names], data_train.value)

        for idx in test_idxs:
            data_test = data[data.video == videos[idx]]
            y_pred = model.predict(data_test.loc[:, names])

            stats["PLCC"][fold].append(pearsonr(y_pred, data_test.value).statistic)
            stats["SRCC"][fold].append(spearmanr(y_pred, data_test.value).statistic)
            stats["KRCC"][fold].append(kendalltau(y_pred, data_test.value).statistic)
            stats["RMSE"][fold].append(np.sqrt(np.mean(np.square(y_pred - data_test.value))))

    return {
        name: np.mean([np.mean(fold) for fold in stat.values()])
        for name, stat in stats.items()
    }


def cross_dataset(train_dataset, test_dataset, model, names):
    train_data = pd.read_csv(train_dataset)
    test_data = pd.read_csv(test_dataset)

    stats = defaultdict(list)

    model.fit(train_data.loc[:, names], train_data.value)

    for video in test_data.video.unique():
        data_test = test_data[test_data.video == video]
        y_pred = model.predict(data_test.loc[:, names])

        stats["PLCC"].append(pearsonr(y_pred, data_test.value).statistic)
        stats["SRCC"].append(spearmanr(y_pred, data_test.value).statistic)
        stats["KRCC"].append(kendalltau(y_pred, data_test.value).statistic)
        stats["RMSE"].append(np.sqrt(np.mean(np.square(y_pred - data_test.value))))

    return {
        name: np.mean(stat)
        for name, stat in stats.items()
    }


if __name__ == "__main__":
    model = Pipeline([
        ("regressor", RandomForestRegressor(n_estimators=400, max_features="sqrt", random_state=8))
    ])

    print("Dataset: dataset_based.csv")
    print("Method\t\t |  PLCC  |  SRCC  |  KRCC  |  RMSE")
    for method, stats in [
        ("based[cv]", cross_validate("dataset_based.csv", model, names)),
        ("based[cd]", cross_dataset("dataset_rsblur.csv", "dataset_based.csv", model, names)),
    ]:
        print(f"{method}\t | {stats['PLCC']:.4f} | {stats['SRCC']:.4f} | {stats['KRCC']:.4f} | {stats['RMSE']:15.4f}")

    print("Dataset: dataset_rsblur.csv")
    print("Method\t\t |  PLCC  |  SRCC  |  KRCC  |  RMSE")
    for method, stats in [
        ("based[cv]", cross_validate("dataset_rsblur.csv", model, names)),
        ("based[cd]", cross_dataset("dataset_based.csv", "dataset_rsblur.csv", model, names)),
    ]:
        print(f"{method}\t | {stats['PLCC']:.4f} | {stats['SRCC']:.4f} | {stats['KRCC']:.4f} | {stats['RMSE']:15.4f}")
