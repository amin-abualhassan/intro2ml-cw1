import argparse
import numpy as np

def load_wifi_dataset(path):
    '''
    parameters:
        path (str): path to the dataset file in .txt format

    functionality:
        Load the dataset from a text file and split it into features (X) and labels (y).

    return:
        tuple: (X, y) where X is the feature array and y is the class/label array
    '''

    arr = np.loadtxt(path)  # load data from text file
    X = arr[:, :-1].astype(float)  # all columns except last = features
    y = arr[:, -1].astype(int)     # last column = labels
    return X, y

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--peek", type=str, help="Path to dataset TXT to preview basic stats.")
    args = ap.parse_args()
    if args.peek:
        X, y = load_wifi_dataset(args.peek)
        print(f"Loaded: X={X.shape}, y={y.shape}, labels={sorted(np.unique(y))}")
        print(f"Feature mins: {X.min(axis=0)}\nFeature maxs: {X.max(axis=0)}\nFeature means: {X.mean(axis=0)}")
