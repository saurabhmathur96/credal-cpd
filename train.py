from datasets import get_dataset
from sklearn.model_selection import StratifiedKFold
import numpy as np
from tqdm import tqdm

for name in ["haberman", "diabetes", "breast-cancer", "thyroid", "heart-disease"]
    df, cardinality, target, synergies, monotonicities = get_dataset("diabetes")
    df = df[[v for (v, s) in monotonicities] + [target]]

    acc = []
    uncertain = []
    X, y = df.drop([target], axis=1).to_numpy(), df[target].to_numpy()
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for (train_index, test_index) in kfold.split(X, y):
        cpd = CredalClassifier(target, [c for c in df.columns if c!=target], cardinality, s0=1, sn=1)
        cpd.fit(X[train_index], y[train_index], [])
        y_pred = cpd.predict(X[test_index])
        y_test = y[test_index]
        acc.append(accuracy_score(y_pred[y_pred != -1], y_test[y_pred!=-1]))
        uncertain.append((y_pred == -1).mean())

    acc = np.array(acc)
    uncertain = np.array(uncertain)
    print (f"acc. = {np.mean(acc):.2f} ± {np.std(acc):.2f}")
    print (f"unc. = {np.mean(uncertain):.2f} ± {np.std(uncertain):.2f}")



    acc = []
    uncertain = []
    X, y = df.drop([target], axis=1).to_numpy(), df[target].to_numpy()
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for (train_index, test_index) in tqdm(kfold.split(X, y)):
        cpd = CredalClassifier(target, [c for c in df.columns if c!=target], cardinality, s0=1, sn=1)
        cpd.fit(X[train_index], y[train_index], monotonicities)
        y_pred = cpd.predict(X[test_index])
        y_test = y[test_index]
        acc.append(accuracy_score(y_pred[y_pred != -1], y_test[y_pred!=-1]))
        uncertain.append((y_pred == -1).mean())

    acc = np.array(acc)
    uncertain = np.array(uncertain)
    print (f"acc. = {np.mean(acc):.2f} ± {np.std(acc):.2f}")
    print (f"unc. = {np.mean(uncertain):.2f} ± {np.std(uncertain):.2f}")