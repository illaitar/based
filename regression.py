import os
import pickle

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import RANSACRegressor
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from metric import *

def comp1(im1, im2):
    return np.sum(cv2.Canny(im1, 150, 255))

def comp2(im1, im2):
    return np.sum(cv2.Canny(im2, 150, 255))

def prepare_dataset(components):
    subj = pd.read_csv(f"subj_{eval_dataset}.csv", index_col=0)

    data = []
    for record in tqdm(subj.itertuples(), total=len(subj)):
        # if (record.method == "real_blur"):
        #     continue
        img1 = cv2.imread(os.path.join(f"crops_{eval_dataset}", record.video, f"{record.method}.png"))
        img2 = cv2.imread(os.path.join(f"crops_{eval_dataset}", record.video, "real_blur.png"))

        values = {"result": record.value}
        for component in components:
            values[component.__name__] = component(img1, img2)

        data.append(values)

    pd.DataFrame(data).to_csv(f"dataset_{eval_dataset}.csv")


def train(data_csv, components):
    data = pd.read_csv(data_csv, index_col=0)

    # model = RANSACRegressor(
    # 220 best
    model = Pipeline([
        ("regressor", RandomForestRegressor(n_estimators=220, random_state=8, criterion="squared_error"))
        # ("regressor", LinearRegression())
    ])
        # ,max_trials=50,min_samples=10,random_state=0)

    names = [component.__name__ for component in components]
    X, y = data.loc[:, names], data.result

    scores = cross_val_score(model, X, y, cv=5, scoring="neg_root_mean_squared_error")
    print("%0.2f rmse with a standard deviation of %0.2f" % (-scores.mean(), scores.std()))

    model.fit(X, y)

    for i, name in enumerate(names):
        print(name, ":", model.named_steps['regressor'].feature_importances_[i])

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)


def regression(img1, img2):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    X = np.array([component(img1, img2) for component in components]).reshape(1, -1)

    return model.predict(X).squeeze()


components = [
    laplac_calc,
    fft_calc,
    optical_calc,
    reblur_calc,
    haff_calc,
    sobel_calc,
    hog_calc,
    lbp_calc,
    gabor_calc,
    ssim_calc,
    # regression,
    haar_calc,
    lpips_calc,
    color_calc,
    tenengrad_calc,
    lapm_calc,
    laple_calc,
    log_calc,
    sharr_calc,
    clache_calc,
    hist_cmp,
    saliency_calc
]


if __name__ == "__main__":
    # eval_dataset = "rsblur"

    # prepare_dataset(components)

    eval_dataset = "based"

    prepare_dataset(components)

    eval_dataset = "based"
    train(f"dataset_{eval_dataset}.csv", components)

# if __name__ == "__main__":
#      # prepare_dataset(comps)
#      data_rsblur = pd.read_csv(f"dataset_rsblur.csv", index_col=0)
#      data_based = pd.read_csv(f"dataset_based.csv", index_col=0)

#      for column in data_based.columns.drop("result"):
#          points_rsblur = data_rsblur.groupby(column).result.mean().to_frame().reset_index()
#          points_based = data_based.groupby(column).result.mean().to_frame().reset_index()
#          points = pd.concat([points_rsblur.assign(dataset="rsblur"), points_based.assign(dataset="based")])
#          points[column] = MinMaxScaler().fit_transform(np.array(points[column]).reshape(-1, 1))
#          sns.jointplot(x=column, y="result", data=points, hue="dataset", alpha=0.5)
#          plt.show()

# if __name__ == "__main__":
#     data_based = pd.read_csv("dataset_based.csv", index_col=0)
#     subj = pd.read_csv(f"subj_based.csv", index_col=0)
#     data_based["video"] = [record.video for record in subj.itertuples()]

#     for column in data_based.columns.drop("result"):
#         g = sns.jointplot(x=column, y="result", data=data_based, hue="metric", alpha=1)
#         # sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1), title='Blya')
#         plt.show()
