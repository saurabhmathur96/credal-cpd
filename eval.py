from model import *
from datasets import get_dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
from os import path
from joblib import load

def accuracy(y_true, y_pred):
    return accuracy_score(y_pred[(y_pred != -1)], y_true[(y_pred!=-1)])

def uncertainty(y_true, y_pred):
    return sum(y_pred==-1)/len(y_true)

def discounted_accuracy(y_true, y_pred):
    t1 = accuracy_score(y_pred[(y_pred != -1)], y_true[(y_pred!=-1)], normalize=False)
    t2 = sum(y_pred==-1)*0.5
    return (t1+t2) / len(y_true)

def f1(y_true, y_pred):
    t1 = accuracy_score(y_pred[y_pred != -1], y_true[y_pred!=-1], normalize=False)
    t2 = sum(y_pred==-1)
    recall = (t1 + t2)/len(y_true) 
    precision = discounted_accuracy(y_true, y_pred)
    print (np.array([recall, precision]).round(4))
    return 2*(precision*recall)/(precision + recall)
metrics = { "accuracy": accuracy, "uncertainty": uncertainty, "discounted_accuracy": discounted_accuracy }

model_dir = "models"
result_dir = "results"
datasets = ["asia", "cancer", "lucas_a", "lucas_b", "lucas_c",
    "haberman", "diabetes", "thyroid", "heart-disease",  "breast-cancer",
    "adni", "rare", "ppd", "numom2b"]
for metric in ["accuracy", "uncertainty", "discounted_accuracy"]:
    for name in datasets:
        print (name)
        df, cardinality, target, synergies, monotonicities = get_dataset(name)
        parents = [v for (v, s) in monotonicities] 
        df = df[parents + [target]]

        models0 = load(path.join(model_dir, f"{name}-0.joblib"))
        models1 = load(path.join(model_dir, f"{name}-1.joblib"))
        models2 = load(path.join(model_dir, f"{name}-2.joblib"))
        models3 = load(path.join(model_dir, f"{name}-3.joblib"))
        # models4 = load(path.join(model_dir, f"{name}-4.joblib"))

        scores0, scores1, scores2, scores3, scores4 = [], [], [], [], []
        X, y = df.drop([target], axis=1).to_numpy(), df[target].to_numpy()
        kfold = StratifiedKFold(n_splits=5)
        for i, (train_index, test_index) in enumerate(kfold.split(X, y)):
            y_true = y[test_index]

            test_data = pd.DataFrame(X[test_index], columns=parents)
            
            p = models0[i].predict_probability(test_data)
            y_pred = (p[f"{target}_1"] >= 0.5).astype(int).to_numpy()
            scores0.append(metrics[metric](y_true, y_pred))

            cpd1 = models1[i]
            y_pred = cpd1.predict(X[test_index])
            scores1.append(metrics[metric](y_true, y_pred))

            cpd2 = models2[i]
            y_pred = cpd2.predict(X[test_index])
            scores2.append(metrics[metric](y_true, y_pred))
            
            cpd3 = models3[i]
            y_pred = cpd3.predict(X[test_index])
            scores3.append(metrics[metric](y_true, y_pred))

            #cpd4 = models4[i]
            #y_pred = cpd4.predict(X[test_index])
            #scores4.append(discounted_accuracy(y_true, y_pred))
        result = pd.DataFrame({
            0: scores0,
            1: scores1,
            2: scores2,
            3: scores3,
            #4: scores4,
        })
        print (result.mean())
        print (result.std())
        result.to_csv(path.join(result_dir, f"{name}-{metric}.csv"), index=False)

            
