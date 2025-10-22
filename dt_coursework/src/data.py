import argparse
import numpy as np

def load_wifi_dataset(path):
    '''
    Input:
        path: path of the file (txt format) to load
    Process:
        Lad the file into 2 arrays:
            X: the dataset features
            y: the dataset class/label (the last column of the dataset) 
    Return:
        Tuple of array of feature (X) and array of class (Y)
    '''
    arr = np.loadtxt(path)
    X = arr[:, :-1].astype(float)
    y = arr[:, -1].astype(int)
    return X, y

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--peek", type=str, help="Path to dataset TXT to preview basic stats.")
    args = ap.parse_args()
    if args.peek:
        X, y = load_wifi_dataset(args.peek)
        print(f"Loaded: X={X.shape}, y={y.shape}, labels={sorted(np.unique(y))}")
        print(f"Feature mins: {X.min(axis=0)}\nFeature maxs: {X.max(axis=0)}\nFeature means: {X.mean(axis=0)}")
