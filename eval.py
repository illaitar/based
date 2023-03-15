import os
from collections import defaultdict

import pandas as pd
import cv2
from pathos.multiprocessing import Pool
from scipy.stats import kendalltau, pearsonr, spearmanr
import numpy as np
from tqdm import tqdm
import warnings

from metric import gabor_calc, sobel_calc, hog_calc, lpb_calc, haff_calc


warnings.filterwarnings("ignore")
eval_dataset = "based"


blur_method = "restormer.png" if eval_dataset == "based" else "real_blur.png"
subjective_table = f"{eval_dataset}.csv"


videos = sorted(os.listdir(f"crops_{eval_dataset}"))
methods = sorted(os.listdir(os.path.join(f"crops_{eval_dataset}", videos[0])))
methods = [method.replace('.png', '') for method in methods]




def correlations(results, corr_func = kendalltau):
    table = pd.read_csv(f"subj_{eval_dataset}.csv", index_col=0)

    correlations = {}

    for video in videos:
        data = defaultdict(dict)
        reference, target = [], []
        for method in methods:

            reference.append(table.loc[((table['video'] == video) & (table['method'] == method))]['value'].values[0])
            target.append(results[video][method])
        correlations[video] = corr_func(reference, target)[0]

    return np.mean([elem for elem in list(correlations.values()) if elem is not np.NaN])


def test_metric(metric):
    results = defaultdict(dict)
    for video in tqdm(videos):
        for method in methods:
            target = cv2.imread(os.path.join(f"crops_{eval_dataset}", video, method + ".png"))
            reference = cv2.imread(os.path.join(f"crops_{eval_dataset}", video, blur_method))
            results[video][method] = metric(target, reference)

    return results


def test_single(args):
    video, method, metric = args
    target = cv2.imread(os.path.join(f"crops_{eval_dataset}", video, method + ".png"))
    reference = cv2.imread(os.path.join(f"crops_{eval_dataset}", video, blur_method))
    value = metric(target, reference)
    return (video, method, value)


def test_metric_mp(metric, num_workers = 1):
    results = defaultdict(lambda: defaultdict(np.float64))
    test = [(video, method, metric) for video in videos for method in methods]
    with Pool(processes=num_workers) as pool:
        out = pool.map(test_single, test)
    for elem in out:
        results[elem[0]][elem[1]] = elem[2]
    return results


if __name__ == "__main__":
    for component in [
        haff_calc,
        sobel_calc,
        hog_calc,
        lpb_calc,
        gabor_calc
    ]:
        v = test_metric_mp(component)
        print(
            f"ktau[{component.__name__}] =",
            correlations(v, kendalltau),
        )
        print(
            f"pearson[{component.__name__}] =",
            correlations(v, pearsonr),
        )
        print(
            f"spearman[{component.__name__}] =",
            correlations(v, spearmanr),
        )
