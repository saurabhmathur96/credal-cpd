from model import *
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from datasets import get_dataset
import pandas as pd


pd.set_option('display.max_colwidth', None)

datasets = ["asia", "cancer", "lucas_a", "lucas_b", "lucas_c",
    "haberman", "diabetes", "thyroid", "heart-disease",  "breast-cancer",
    "adni", "rare", "ppd", "numom2b"]
rows = []
for name in datasets:
    df, cardinality, target, synergies, monotonicities = get_dataset(name)
    
    parents = [v for (v, s) in monotonicities] 
    df = df[parents + [target]]

    rows.append([name, len(df), target, ",".join([f"{parent}$^+$" if sign==+1 else f"{parent}$^-$"  for (parent, sign) in monotonicities]) ])

names = ["name", "$|D|$", "Y", "X"]
print (pd.DataFrame(rows, columns=names).to_latex(index=False,escape=False))