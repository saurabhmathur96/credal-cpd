from os import path, listdir
import pandas as pd


results_dir = "results"
metrics = ["accuracy", "uncertainty", "discounted_accuracy"]
datasets = ["asia", "cancer", "lucas_a", "lucas_b", "lucas_c",
    "haberman", "diabetes", "thyroid", "heart-disease",  "breast-cancer",
    "adni", "rare", "ppd", "numom2b"]
for metric in metrics:
    rows = []
    for name in datasets:
        file_path = path.join(results_dir, f"{name}-{metric}.csv")
        df = pd.read_csv(file_path)
        row =  ([name]+[f'{df[str(i)].mean().round(3):.3f} ± {df[str(i)].std().round(2):.2f}'
                for i in [0,1,2,3]])
        rows.append(row)
    names = ["name", "BN", "IDM", "IDM + constrained prior", "IDM + constrained posterior"]
    print (pd.DataFrame(rows, columns=names).to_latex(index=False))