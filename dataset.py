import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from os import path 

raw_data_dir = "raw_data"

class Dataset:
    def __init__(self, frame, cardinality, names, monotonicities):
        self.frame = frame
        self.cardinality = cardinality
        self.names = names
        self.monotonicities = monotonicities
    
    def add_noise(self, fraction: float = 0.2):
        n = len(self.frame)
        sample = pd.DataFrame({ 
            name: np.random.randint(0, self.cardinality[name], size = int(n*fraction))
            for name in self.names 
        }).astype(int)
        
        return Dataset(sample.append(self.frame),
                       self.cardinality,
                       self.names,
                       self.monotonicities)

    def compute_counts(self, variable, parents):
        value_counts = self.frame[[variable, *parents]].value_counts()
        cardinality = self.cardinality
        parent_card = [cardinality[v] for v in parents]
        variable_card = cardinality[variable]
        counts = np.zeros((np.prod(parent_card), variable_card))
        for i, value in enumerate(np.ndindex(*parent_card)):
            for j in range(variable_card):
                counts[i, j] = value_counts.get((j, *value), 0)
        return counts
        
def read_dataset(name: set):
    if name == "heart_disease":
        names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca",
             "thal", "num"]
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        frame = pd.read_csv(url, names=names, na_values="?")
        frame = frame[["sex", "age", "trestbps", "chol", "fbs", "num"]].dropna()
        frame["unhealthy"] = (frame.num.astype(int) != 0).astype(int)
        frame.drop(['num'], axis=1, inplace=True)
        frame["chol"] = (frame["chol"] > 240) 
        frame["trestbps"] = pd.cut(frame["trestbps"], [0, 120, 140, np.inf], labels = np.arange(3))
        frame["age"] = pd.cut(frame["age"], [0, 40, 60, np.inf], labels = np.arange(3))
        frame = frame.astype(int)
        monotonicities = { "unhealthy": [("age", +1), ("trestbps", +1), ("chol", +1), ("fbs", +1), ("sex", +1)] }
        cardinality = { v: frame[v].max()+1 for v in frame.columns }
    elif name == "diabetes":
        frame = pd.read_csv(path.join(raw_data_dir, "diabetes", "diabetes.csv"))
        frame.iloc[:, [1,2,3,4,5,6,7]] = frame.iloc[:, [1,2,3,4,5,6,7]].replace(0, np.NaN)
        frame = frame[["age", "mass", "pedi", "preg", "class"]].dropna()
        frame["class"] = (frame["class"] == "tested_positive").astype(int)
        frame["preg"] = pd.cut(frame["preg"], [-np.inf, 2, 6, np.inf], labels = np.arange(3))
        frame["age"] = pd.cut(frame["age"], [-np.inf, 30, 40, np.inf], labels = np.arange(3))
        frame["mass"] = pd.cut(frame["mass"], [-np.inf, 22.8, 26.8, 33.6, 35.6, np.inf], labels = np.arange(5))
        frame["pedi"] = pd.cut(frame["pedi"], [-np.inf, .244, .525, .805, 1.11, np.inf], labels = np.arange(5)) 
        frame = frame.astype(int)
        monotonicities = { "class": [("preg", +1), ("pedi", +1), ("age", +1), ("mass", +1)] }
        cardinality = { v: frame[v].max()+1 for v in frame.columns }
    elif name == "ljubljana":
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data"
        names = ["class", "age", "menopause", "size", "invnodes", "nodecaps", "degmalig", "breast", "quad", "irradiat"]
        frame = pd.read_csv(url, names=names, na_values="?")
        frame = frame[["age", "menopause", "size", "degmalig", "irradiat", "class"]].dropna()
        frame["class"].replace(dict(zip(["no-recurrence-events", "recurrence-events"], range(2))), inplace=True)
        frame["age"].replace(
        dict(zip(['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99'], range(9))),
        inplace=True)
        frame["menopause"].replace(dict(zip(["premeno", "lt40", "ge40"], range(3))), inplace=True)
        frame["size"].replace(dict(
        zip(['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59'],
            range(12))), inplace=True)
        frame["degmalig"] -= 1
        frame["irradiat"].replace({"no": 0, "yes": 1}, inplace=True)
        k = 4
        for name in ["age", "size"]:
            frame[name] = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='kmeans') \
                        .fit_transform(frame[name].to_numpy().reshape(-1, 1)) \
                        .flatten().astype(int)
        frame = frame.astype(int)
        monotonicities = { "class": [("age", +1), ("menopause", +1), ("size", +1), ("degmalig", +1), ("irradiat", -1)] }
        cardinality = { v: frame[v].max()+1 for v in frame.columns }
    elif name == "wisconsin":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
        names = ["id", "clumpthick", "cellsize", "cellshape", "adhesion", "epitsize", "barenuc", "blandchr", "normnuc", "mitoses", "malignant"]
        frame = pd.read_csv(url, names = names, index_col = 0, na_values="?").dropna()
        frame["malignant"] = (frame["malignant"] == 4).astype(int)

        k = 5
        for name in frame.columns[:-1]:
            frame[name] = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='kmeans') \
                .fit_transform(frame[name].to_numpy().reshape(-1, 1)) \
                .flatten().astype(int)
        frame = frame.astype(int)
        monotonicities = { n: [("malignant", +1)] for n in frame.columns[:-1] }
        cardinality = {'malignant':2, **{ v: 5 for v in frame.columns[:-1] } }

    else:
        raise ValueError("Invalid name")
    
    frame = frame.astype(int)
    
    tr, te= train_test_split(frame, stratify = frame.iloc[:, -1], test_size=0.5, random_state=123)
    names = frame.columns.tolist()
    train = Dataset(tr, cardinality, names, monotonicities)
    test = Dataset(te, cardinality, names, monotonicities)
    
    return train, test
