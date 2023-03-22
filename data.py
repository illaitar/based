import os

import cv2
import pandas as pd
from tqdm import tqdm
from pathos.multiprocessing import Pool

from metric import *
from model import deblur_compare


def prepare_row(record):
    img1 = cv2.imread(os.path.join(f"crops_{dataset}", record.video, f"{record.method}.png"))
    img2 = cv2.imread(os.path.join(f"crops_{dataset}", record.video, "real_blur.png"))

    record = {
        "method": record.method,
        "value": record.value,
        "video": record.video,
    }

    for component in components:
        result = component(img1, img2)
        if type(result) is list:
            for i, elem in enumerate(result):
                record[component.__name__ + str(i)] = elem
        else:
            record[component.__name__] = result

    return record


components = [
    # laplac_calc,
    # fft_calc,
    # # optical_calc,
    # # reblur_calc,
    # haff_calc,
    # sobel_calc,
    hog_calc,
    # lbp_calc,
    # gabor_calc,
    # ssim_calc,
    # # regression,
    # haar_calc,
    # # lpips_calc,
    # color_calc,
    # tenengrad_calc,
    # lapm_calc,
    # laple_calc,
    # log_calc,
    # sharr_calc,
    # # # clache_calc,
    # hist_cmp,
    # saliency_calc,


    # fft2_calc,



    # ssim_blurriness_metric,
    # vif_blurriness_metric,
    # vollath_blurriness_metric
    # wavelet_blurriness_metric,
    # fft3
]

names = [
    component.__name__
    for component in components
]


if __name__ == "__main__":
    for dataset in [
        "based",
        "rsblur",
    ]:
        subj = pd.read_csv(f"subj_{dataset}.csv", index_col=0)

        with Pool(8) as pool:
            records = list(tqdm(pool.imap(prepare_row, subj.itertuples()), total=len(subj)))

        pd.DataFrame(records).to_csv(f"dataset_{dataset}.csv", index=False)
