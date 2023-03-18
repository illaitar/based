import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import cv2
from pathos.multiprocessing import Pool
from scipy.stats import kendalltau, pearsonr, spearmanr
import numpy as np
from tqdm import tqdm
import warnings

from metric import *
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
            reference = cv2.imread(os.path.join(f"crops_{eval_dataset}", video, "real_blur.png"))
            results[video][method] = metric(target, reference)

    return results


def test_single(args):
    video, method, metric = args
    target = cv2.imread(os.path.join(f"crops_{eval_dataset}", video, method + ".png"))
    reference = cv2.imread(os.path.join(f"crops_{eval_dataset}", video, "real_blur.png"))
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


def get_examples(better, worse):
    examples = []
    for video in videos:
        table = pd.read_csv(f"subj_{eval_dataset}.csv", index_col=0)
        subj = {}
        for i, method in enumerate(methods):
            subj[method] =  table.loc[((table['video'] == video) & (table['method'] == method))]['value'].values[0]

        for method1 in methods:
            for method2 in methods:
                if method1 == method2:
                    continue
                if ((subj[method1] >  subj[method2]) and (better[video][method1] > better[video][method2]) and (worse[video][method1] < worse[video][method2])):
                    examples.append((method1, method2, video, np.round(better[video][method1][0], decimals=4), np.round(better[video][method2][0], decimals=4), np.round(worse[video][method1], decimals=4), np.round(worse[video][method2], decimals=4)))
    return examples


def show_examples(examples, name_better, name_worse):
    for i, example in enumerate(examples):
        method1, method2, video, better1, better2, worse1, worse2 = example

        im1 = cv2.imread(f"./crops_{eval_dataset}/{video}/{method1}.png")
        im2 = cv2.imread(f"./crops_{eval_dataset}/{video}/{method2}.png")

        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

        figure, axis = plt.subplots(1, 2)
        figure.set_size_inches(9, 5)

        axis[0].imshow(im1)
        axis[0].set_title(f"{name_better}:{round(better1, 4)}; {name_worse}:{round(worse1, 4)}")

        axis[1].imshow(im2)
        axis[1].set_title(f"{name_better}:{round(better2, 4)}; {name_worse}:{round(worse2, 4)}")
        os.makedirs("./examples", exist_ok = True)
        os.makedirs(f"./examples/{eval_dataset}", exist_ok = True)
        figure.savefig(f"./examples/{eval_dataset}/{i}.png")   # save the figure to file
        plt.close(figure)


def metrics_correlations(results1, results2, metric1, metric2, videos):
    r1 = []
    r2 = []
    for video in videos:
        r1.append(
                    np.mean(list(results1[video].values()))
                 )
        r2.append(
                    np.mean(list(results2[video].values()))
                 )
    corr = kendalltau(r1, r2)[0]
    return corr


def metrics_correlations_all(results, metrics):
    data = {}
    data["metric"] = [metric.__name__  for metric in metrics]
    for i1, metric in enumerate(metrics):
        tmp = []
        for i2, metric2 in enumerate(metrics):
            if metric.__name__ == metric2.__name__:
                tmp.append(1)
            else:
                tmp.append(metrics_correlations(results[i1], results[i2], metric, metric2, videos))
        data[metric.__name__] = tmp
    corrs = pd.DataFrame(data)
    corrs.to_csv(f"metrics_corrs_{eval_dataset}.csv")


if __name__ == "__main__":
    for eval_dataset in [
        "based",
        "rsblur"
    ]:
        subjective_table = f"{eval_dataset}.csv"

        videos = sorted(os.listdir(f"crops_{eval_dataset}"))
        methods = sorted(os.listdir(os.path.join(f"crops_{eval_dataset}", videos[0])))
        methods = [method.replace(".png", "") for method in methods]


        print("\t\t\tDataset:", eval_dataset)
        print("Name\t\t | Time, s |  PLCC  |  SRCC  |  KRCC  |")
        for component in [
            # laplac_calc,
            # fft_calc,
            # optical_calc,
            # reblur_calc,
            # haff_calc,
            # sobel_calc,
            # hog_calc,
            lbp_calc,
            # gabor_calc,
            # ssim_calc,
            # regression,
            # lpips_calc,
            # color_calc
        ]:
            tic = time.time()
            v = test_metric_mp(component)
            toc = time.time()

            plcc = correlations(v, pearsonr)
            srcc = correlations(v, spearmanr)
            krcc = correlations(v, kendalltau)

            print(f"{component.__name__}\t | {int(toc - tic): 7d} | {plcc:.4f} | {srcc:.4f} | {krcc:.4f} |")



        print()

        # components = [
        #     laplac_calc,
        #     fft_calc,
        #     optical_calc,
        #     reblur_calc,
        #     haff_calc,
        #     sobel_calc,
        #     hog_calc,
        #     lbp_calc,
        #     gabor_calc,
        #     ssim_calc,
        #     regression,
        #     lpips_calc
        # ]

        # rs = []
        # for component in components:
        #     rs.append(test_metric_mp(component))
        # metrics_correlations_all(rs, components)
