import os
from collections import defaultdict

import pandas as pd
import cv2
from scipy.stats import kendalltau
import numpy as np
from tqdm import tqdm
import warnings

from metric import gabor_calc, sobel_calc, hog_calc, lpb_calc


warnings.filterwarnings("ignore")
eval_dataset = "rsblur"


blur_method = "restormer.png" if eval_dataset == "based" else "real_blur.png"
subjective_table = f"{eval_dataset}.csv"


videos = sorted(os.listdir(f"crops_{eval_dataset}"))
videos.remove(".DS_Store")
methods = sorted(os.listdir(os.path.join(f"crops_{eval_dataset}", videos[0])))
methods = [method.replace('.png', '') for method in methods]
if eval_dataset == "based":
    methods.remove("gt.png")



def ktau(results):
    table = pd.read_csv(f"subj_{eval_dataset}.csv", index_col=0)

    correlations = {}

    for video in videos:
        data = defaultdict(dict)
        reference, target = [], []
        for method in methods:
            reference.append(table.loc[((table['video'] == video) & (table['method'] == method))]['value'].values[0])
            target.append(results[video][method])
        correlations[video] = kendalltau(reference, target)[0]

    return np.mean([elem for elem in list(correlations.values()) if elem is not np.NaN])


def test_metric(metric):
    results = defaultdict(dict)
    for video in tqdm(videos):
        for method in methods:
            target = cv2.imread(os.path.join(f"crops_{eval_dataset}", video, method + ".png"))
            reference = cv2.imread(os.path.join(f"crops_{eval_dataset}", video, blur_method))
            results[video][method] = metric(target, reference)

    return results


if __name__ == "__main__":
    for component in [
        sobel_calc,
        hog_calc,
        lpb_calc,
        gabor_calc,
    ]:
        print(
            f"ktau[{component.__name__}] =",
            ktau(test_metric(component)),
        )
