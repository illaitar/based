import math
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau, ConstantInputWarning


warnings.filterwarnings(
    action="ignore",
    category=ConstantInputWarning
)


def evaluate(dataset, method):
    data = pd.read_csv(dataset)
    videos = data.video.unique()

    stats = defaultdict(list)

    for video in videos:
        data_test = data[data.video == video]

        stats["PLCC"].append(pearsonr(data_test[method], data_test.value).statistic)
        stats["SRCC"].append(spearmanr(data_test[method], data_test.value).statistic)
        stats["KRCC"].append(kendalltau(data_test[method], data_test.value).statistic)
        stats["RMSE"].append(np.sqrt(np.mean(np.square(data_test[method] - data_test.value))))

    return {
        name: np.abs(np.mean([0 if math.isnan(x) else x for x in stat]))
        for name, stat in stats.items()
    }


if __name__ == "__main__":
    for dataset in [
        "dataset_based.csv",
        "dataset_rsblur.csv"
    ]:
        print("Dataset:", dataset)
        print("Method\t\t |  PLCC  |  SRCC  |  KRCC  |  RMSE")

        for method in [
            "haff_calc",
            "sobel_calc",
            "hog_calc",
            "lbp_calc"
        ]:
            stats = evaluate(
                dataset=dataset,
                method=method
            )

            print(f"{method}\t | {stats['PLCC']:.4f} | {stats['SRCC']:.4f} | {stats['KRCC']:.4f} | {stats['RMSE']:15.4f}")
