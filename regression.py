import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pickle
import os


def train(data_csv, components, save_path = "./"):
    data = pd.read_csv(os.path.join(save_path, data_csv))
    labels = [i for i in data.columns if i not in components and i != 'result']
    data.drop(columns = labels,  axis=1, inplace=True)
    model = LinearRegression(fit_intercept=False)
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

if __name__ == "__main__":
    train('dataset.csv', ['gabore', 'hog', 'lbp', 'sobel'])
