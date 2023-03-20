import os

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kendalltau


eval_dataset = "based"
videos = pd.read_csv(f"subj_{eval_dataset}.csv").video.unique()
methods = pd.read_csv(f"subj_{eval_dataset}.csv").method.unique()


def get_examples(better, worse):
    examples = []
    for video in videos:
        table = pd.read_csv(f"subj_{eval_dataset}.csv", index_col=0)
        subj = {}
        for i, method in enumerate(methods):
            subj[method] = table.loc[((table['video'] == video) & (table['method'] == method))]['value'].values[0]

        for method1 in methods:
            for method2 in methods:
                if method1 == method2:
                    continue
                if ((subj[method1] > subj[method2]) and (better[video][method1] > better[video][method2]) and (worse[video][method1] < worse[video][method2])):
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
        os.makedirs("./examples", exist_ok=True)
        os.makedirs(f"./examples/{eval_dataset}", exist_ok=True)
        figure.savefig(f"./examples/{eval_dataset}/{i}.png")   # save the figure to file
        plt.close(figure)


def metrics_correlations(results1, results2, videos):
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
                tmp.append(metrics_correlations(results[i1], results[i2], videos))
        data[metric.__name__] = tmp
    corrs = pd.DataFrame(data)
    corrs.to_csv(f"metrics_corrs_{eval_dataset}.csv")


data_rsblur = pd.read_csv(f"dataset_rsblur.csv", index_col=0)
data_based = pd.read_csv(f"dataset_based.csv", index_col=0)

for column in data_based.columns.drop("result"):
    points_rsblur = data_rsblur.groupby(column).result.mean().to_frame().reset_index()
    points_based = data_based.groupby(column).result.mean().to_frame().reset_index()
    points = pd.concat([points_rsblur.assign(dataset="rsblur"), points_based.assign(dataset="based")])

    sns.jointplot(x=column, y="result", data=points, hue="dataset", alpha=0.5)
    plt.show()

data_based = pd.read_csv("dataset_based.csv", index_col=0)
subj = pd.read_csv(f"subj_based.csv", index_col=0)
data_based["video"] = [record.video for record in subj.itertuples()]

for column in data_based.columns.drop("result"):
    g = sns.jointplot(x=column, y="result", data=data_based, hue="metric", alpha=1)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1), title='Blya')
    plt.show()
