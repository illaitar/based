# BASED: Benchmarking, Analysis, and Structural Estimation of Deblurring

This repo contains subjective ranking results on BASED and RSBlur datasets
along with a blur detection metric.

## Installation

```shell
git clone https://github.com/illaitar/based.git
cd based
pip install -qr requirements.txt
```

## Development

Run `data.py` to run metrics (components) and save results to csv files.
Then use `eval.py` to calculate correlations of these components or `train.py` to get
correlations on cross-validation (CV) and cross-dataset (CD) splits.

```shell
python data.py
python eval.py
python train.py
```

## See Also

* [MSU Video Deblurring Benchmark 2022](https://videoprocessing.ai/benchmarks/deblurring.html)
