import os
import pickle

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from metric import *


def prepare_dataset(components):
    subj = pd.read_csv(f"subj_{eval_dataset}.csv", index_col=0)

    data = []
    for record in tqdm(subj.itertuples(), total=len(subj)):
        img1 = cv2.imread(os.path.join(f"crops_{eval_dataset}", record.video, f"{record.method}.png"))
        img2 = cv2.imread(os.path.join(f"crops_{eval_dataset}", record.video, f"{blur_method}.png"))

        values = {"result": record.value}
        for component in components:
            values[component.__name__] = component(img1, img2)

        data.append(values)

    pd.DataFrame(data).to_csv(f"dataset_{eval_dataset}.csv")


def train(data_csv, components):
    data = pd.read_csv(data_csv, index_col=0)

    model = Pipeline([
        ("regressor", RandomForestRegressor(n_estimators=160, random_state=8, criterion="squared_error"))
    ])

    names = [component.__name__ for component in components]
    X, y = data.loc[:, names], data.result

    scores = cross_val_score(model, X, y, cv=5, scoring="neg_root_mean_squared_error")
    print("%0.2f rmse with a standard deviation of %0.2f" % (-scores.mean(), scores.std()))

    model.fit(X, y)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)


def regression(img1, img2):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    X = np.array([component(img1, img2) for component in components]).reshape(1, -1)

    return model.predict(X).squeeze()


components = [
    sobel_calc,
    hog_calc,
    lbp_calc,
    ssim_calc,
    gabor_calc,
    reblur_calc,
    fft_calc,
    optical_calc
]


if __name__ == "__main__":
    eval_dataset = "based"
    blur_method = "restormer" if eval_dataset == "based" else "real_blur"

    prepare_dataset(components)

    train(f"dataset_{eval_dataset}.csv", components)


if __name__ == "__main__":
    data_rsblur = pd.read_csv(f"dataset_rsblur.csv", index_col=0)
    data_based = pd.read_csv(f"dataset_based.csv", index_col=0)

    for column in data_based.columns.drop("result"):
        points_rsblur = data_rsblur.groupby(column).result.mean().to_frame().reset_index()
        points_based = data_based.groupby(column).result.mean().to_frame().reset_index()
        points = pd.concat([points_rsblur.assign(dataset="rsblur"), points_based.assign(dataset="based")])
        points[column] = MinMaxScaler().fit_transform(np.array(points[column]).reshape(-1, 1))
        sns.scatterplot(x=column, y="result", data=points, hue="dataset", alpha=0.5)
        plt.show()
