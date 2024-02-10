from model import *
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from datasets import get_dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
from os import path
from joblib import dump
import copy 

model_dir = "models/"
datasets = ["asia", "cancer", "lucas_a", "lucas_b", "lucas_c",
    "haberman", "diabetes", "thyroid", "heart-disease",  "breast-cancer",
    "adni", "rare", "ppd", "numom2b"]
for name in datasets: 
    print (name)
    df, cardinality, target, synergies, monotonicities = get_dataset(name)
    print (cardinality)
    parents = [v for (v, s) in monotonicities] 
    df = df[parents + [target]]

    epsilon = 0.01 if name in ["cancer", "asia", "lucas_a", "lucas_b", "lucas_c"] else 0.001

    models0, models1,models2,models3,models4 = [], [], [], [], []
    X, y = df.drop([target], axis=1).to_numpy(), df[target].to_numpy()
    kfold = StratifiedKFold(n_splits=5)
    gen = np.random.default_rng(seed = 0)
    for i, (train_index_, test_index) in tqdm(enumerate(kfold.split(X, y)), total=5):
        if name in ["cancer", "asia", "lucas_a", "lucas_b", "lucas_c"]:
            train_index = gen.choice(train_index_, size=50, replace=False)
        else:
            train_index = train_index_

        cpd0 = BayesianNetwork()
        cpd0.add_nodes_from(df.columns.tolist())
        cpd0.add_edges_from([(parent, target) for parent in parents])
        train_data = pd.DataFrame(X[train_index], columns=parents)
        train_data[target] = y[train_index]
        cpd0.fit(train_data, estimator=BayesianEstimator, prior_type="dirichlet", pseudo_counts=2, state_names={
            k: list(range(v))
            for k, v in cardinality.items()
        })
        models0.append(copy.deepcopy(cpd0))
        
        
        cpd1 = CredalClassifier(target, [c for c in df.columns if c!=target], cardinality, s0=2, sn=2)
        cpd1.fit(X[train_index], y[train_index], [], epsilon)
        models1.append(copy.deepcopy(cpd1))
        
        
        cpd2 = CredalClassifier(target, [c for c in df.columns if c!=target], cardinality, s0=2, sn=2, prior_constraint=True)
        cpd2.fit(X[train_index], y[train_index], monotonicities, epsilon)
        models2.append(copy.deepcopy(cpd2))

        cpd3 = CredalClassifier(target, [c for c in df.columns if c!=target], cardinality, s0=2, sn=2)
        cpd3.fit(X[train_index], y[train_index], monotonicities, epsilon)
        models3.append(copy.deepcopy(cpd3))
        

    dump(models0, path.join(model_dir, f"{name}-0.joblib"))
    dump(models1, path.join(model_dir, f"{name}-1.joblib"))
    dump(models2, path.join(model_dir, f"{name}-2.joblib"))
    dump(models3, path.join(model_dir, f"{name}-3.joblib"))