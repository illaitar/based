import os

import cv2
import pandas as pd
from tqdm import tqdm
from pathos.multiprocessing import Pool

from metric import *


def prepare_row(record):
    img1 = cv2.imread(os.path.join(f"crops_{dataset}", record.video, f"{record.method}.png"))
    img2 = cv2.imread(os.path.join(f"crops_{dataset}", record.video, "real_blur.png"))

    record = {
        "method": record.method,
        "value": record.value,
        "video": record.video,
    }

    for component in components:
        record[component.__name__] = component(img1, img2)

    return record


components = [
    haff_calc,
    sobel_calc,
    hog_calc,
    lbp_calc,
]

names = [
    component.__name__
    for component in components
]


if __name__ == "__main__":
    for dataset in [
        "based",
        "rsblur"
    ]:
        subj = pd.read_csv(f"subj_{dataset}.csv", index_col=0)

        with Pool(8) as pool:
            records = list(tqdm(pool.imap(prepare_row, subj.itertuples()), total=len(subj)))

        pd.DataFrame(records).to_csv(f"dataset_{dataset}.csv", index=False)
