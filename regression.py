import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
from tqdm import tqdm
import cv2
import numpy as np


from metric import gabor_calc, sobel_calc, hog_calc, lbp_calc, haff_calc

eval_dataset = "rsblur"

blur_method = "restormer.png" if eval_dataset == "based" else "real_blur.png"
subjective_table = f"{eval_dataset}.csv"

videos = sorted(os.listdir(f"crops_{eval_dataset}"))
methods = sorted(os.listdir(os.path.join(f"crops_{eval_dataset}", videos[0])))
methods = [method.replace(".png", "") for method in methods]


def prepare_dataset(components, save_path="./"):
    base_path = f"./crops_{eval_dataset}"
    combinations = [(video, method) for video in videos for method in methods if method != blur_method.split('.png')[0]]
    data = []
    table = pd.read_csv(f"subj_{eval_dataset}.csv", index_col=0)
    for combination in tqdm(combinations):
        video, method = combination
        target = cv2.imread(os.path.join(base_path, video, f"{method}.png"))
        reference = cv2.imread(os.path.join(base_path, video, f"{blur_method}"))
        result = table.loc[((table['video'] == video) & (table['method'] == method))]['value'].values[0]
        values = {}
        for component in components:
            values[component.__name__] = component(target, reference)
        values['result'] = result
        data.append(values)

    out = pd.DataFrame(data)
    out.to_csv(os.path.join(save_path, f"./dataset_{eval_dataset}.csv"))


def train(data_csv, components, save_path = "./"):
    components = [component.__name__ for component in components]
    data = pd.read_csv(os.path.join(save_path, data_csv))
    labels = [i for i in data.columns if i not in components and i != 'result']
    data.drop(columns = labels,  axis=1, inplace=True)
    model = LinearRegression(fit_intercept=False)
    # model = RandomForestRegressor(n_estimators = 150, random_state = 0)
    X = data.iloc[:, 0:-1].values
    y = data.iloc[:, -1].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    model.fit(X_train, y_train)
    for elem in zip(components, model.coef_):
        print(f"{elem[0]}:{elem[1]}")
    preds_valid = model.predict(X_test)
    score_valid = mean_absolute_error(y_test, preds_valid)
    print("MAE: ",score_valid)
    with open(os.path.join(save_path, ('scaler.pkl')), 'wb') as fid:
        pickle.dump(scaler, fid)
    with open(os.path.join(save_path, ('model.pkl')), 'wb') as fid:
        pickle.dump(model, fid)


def regression(blur, deblur, path="./"):
    with open(os.path.join(path, 'scaler.pkl'), 'rb') as fid:
        scaler = pickle.load(fid)
    with open(os.path.join(path, 'model.pkl'), 'rb') as fid:
        model = pickle.load(fid)
        values = {}
    values['gabore'] = gabor_calc(deblur, blur)
    values['sobel'] = sobel_calc(deblur, blur)
    values['hog'] = hog_calc(deblur, blur)
    values['lbp'] = lbp_calc(deblur, blur)
    values['haff'] = haff_calc(deblur, blur)
    values = list(values.values())
    values = np.array(values).reshape(1, -1)
    values = scaler.transform(values)
    return model.predict(values)


if __name__ == "__main__":
    # prepare_dataset([gabor_calc, sobel_calc, hog_calc, lbp_calc, haff_calc])
    train(f'dataset_{eval_dataset}.csv', [gabor_calc, sobel_calc, hog_calc, lbp_calc, haff_calc])
