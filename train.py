from collections import defaultdict

import numpy as np
from numpy.random.mtrand import RandomState
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

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


if __name__ == "__main__":
    for dataset in [
        "dataset_based.csv",
        "dataset_rsblur.csv"
    ]:
        print("Dataset:", dataset)
        print("Method\t\t |  PLCC  |  SRCC  |  KRCC  |  RMSE")

        stats = cross_validate(
            dataset=dataset,
            model=Pipeline([
                ("regressor",RandomForestRegressor(n_estimators=400, max_features='sqrt', random_state=8))
            ]),
            names=names
        )

        print(f"based\t\t | {stats['PLCC']:.4f} | {stats['SRCC']:.4f} | {stats['KRCC']:.4f} | {stats['RMSE']:15.4f}")
