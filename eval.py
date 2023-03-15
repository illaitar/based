import os
import time
from collections import defaultdict

import pandas as pd
import cv2
from pathos.multiprocessing import Pool
from scipy.stats import kendalltau, pearsonr, spearmanr
import numpy as np
from tqdm import tqdm
import warnings

from metric import gabor_calc, sobel_calc, hog_calc, lbp_calc, haff_calc
from regression import regression


warnings.filterwarnings("ignore")


def correlations(results, corr_func):
    table = pd.read_csv(f"subj_{eval_dataset}.csv", index_col=0)

    data = {}

    for video in videos:
        reference, target = [], []
        for method in methods:
            reference.append(table.loc[((table['video'] == video) & (table['method'] == method))]['value'].values[0])
            target.append(results[video][method])
        data[video] = corr_func(reference, target)[0]

    return np.mean([elem for elem in list(data.values()) if elem is not np.NaN])


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
    return video, method, value


def test_metric_mp(metric, num_workers=8):
    results = defaultdict(lambda: defaultdict(np.float64))
    test = [(video, method, metric) for video in videos for method in methods]
    with Pool(processes=num_workers) as pool:
        out = pool.map(test_single, test)
    for elem in out:
        results[elem[0]][elem[1]] = elem[2]
    return results


if __name__ == "__main__":
    for eval_dataset in [
        "based",
        "rsblur"
    ]:
        blur_method = "restormer.png" if eval_dataset == "based" else "real_blur.png"
        subjective_table = f"{eval_dataset}.csv"

        videos = sorted(os.listdir(f"crops_{eval_dataset}"))
        methods = sorted(os.listdir(os.path.join(f"crops_{eval_dataset}", videos[0])))
        methods = [method.replace(".png", "") for method in methods]

        print("\t\t\tDataset:", eval_dataset)
        print("Name\t\t | Time, s |  PLCC  |  SRCC  |  KRCC  |")
        for component in [
            haff_calc,
            sobel_calc,
            hog_calc,
            lbp_calc,
            gabor_calc,
            regression
        ]:
            tic = time.time()
            v = test_metric_mp(component)
            toc = time.time()

            plcc = correlations(v, pearsonr)
            srcc = correlations(v, spearmanr)
            krcc = correlations(v, kendalltau)

            print(f"{component.__name__}\t | {int(toc - tic): 7d} | {plcc:.4f} | {srcc:.4f} | {krcc:.4f} |")
        print()
