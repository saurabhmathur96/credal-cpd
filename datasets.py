import os
from pgmpy.readwrite import BIFReader
import pandas as pd 
import numpy as np 


bn_path = "bns"
base = "raw-data"
def get_dataset(name: str):
    if name == "haberman":
        path = os.path.join(base, "haberman.data")
        names = ["age", "year", "nodes", "survive"]
        df = pd.read_csv(path, names=names)
        df.survive = (df.survive == 1).astype(int)
        df.nodes = pd.cut(df.nodes, [-np.inf, 10, 30, np.inf], labels = np.arange(3))
        df.year = pd.cut(df.year, [-np.inf, 60, 64, np.inf], labels = np.arange(3))
        df.age = pd.cut(df.age, [-np.inf, 40, 60, np.inf], labels = np.arange(3))
        cardinality = {
            "survive":2,
            "nodes": 3,
            "year": 3,
            "age": 3
        }
        target = "survive"
        monotonicities = [
            ("nodes", -1),
            ("year", +1),
            ("age", -1)
        ]
        synergies = []
        return df, cardinality, target, synergies, monotonicities

    elif name == "breast-cancer":
        path = os.path.join(base, "breast-cancer.data")
        columns = ["recurrence", "age", "menopause", "tumor_size", "inv_nodes",
        "node_caps", "deg_malig", "breast", "breast_quad", "irradiat"]
        df = pd.read_csv(path, na_values="?", names=columns).dropna()
        df.recurrence = (df.recurrence == "recurrence-events")
        df.age.replace({
            "10-19":0, "20-29":0, "30-39":0,
            "40-49":1, "50-59":1, "60-69":1,
            "70-79":2, "80-89":2, "90-99":2 
        }, inplace=True)
        df.menopause = (df.menopause != "premeno")
        df.tumor_size.replace({
            "0-4":0, "5-9":0, "10-14":0, 
            "15-19":1, "20-24":1, "25-29":1, 
            "30-34":2, "35-39":2, "40-44":2, 
            "45-49":3, "50-54":3, "55-59":3
        }, inplace=True)
        df.inv_nodes.replace({
            "0-2":0, "3-5":0, "6-8":0, 
            "9-11":1, "12-14":1, "15-17":1, 
            "18-20":2, "21-23":2, "24-26":2, 
            "27-29":3, "30-32":3, "33-35":3, "36-39":3
        }, inplace=True)
        df.node_caps = (df.node_caps == "yes")
        df.breast = (df.breast == "left")
        df.breast_quad.replace({
            "left_up":0, "left_low":1, "right_up":2, 
            "right_low":3, "central":4
        }, inplace=True)
        df.irradiat = (df.irradiat == "yes")
        df = df.astype(int)
        cardinality = { column: df[column].max() + 1 for column in df.columns }
        target = "recurrence"
        synergies = [("age", "menopause", +1), ("irradiat", "deg_malig", -1), ("irradiat", "tumor_size", -1)]
        monotonicities = [("age", +1), ("menopause", +1), ("deg_malig", +1), ("tumor_size", +1), ("irradiat", -1)]
        return df, cardinality, target, synergies, monotonicities
    
    elif name == "diabetes":
        path = os.path.join(base, "diabetes.csv")
        df = pd.read_csv(path)
        for column in df.columns[1:8]:
            df[column].replace(0, np.NaN, inplace=True)
        
        df = df.dropna()
        df.Age = pd.cut(df.Age, [-np.inf, 30, 40, np.inf], labels = np.arange(3))
        df.BMI = pd.cut(df.BMI, [-np.inf, 25, 30, np.inf], labels = np.arange(3))
        df.BloodPressure = pd.cut(df.BloodPressure, [-np.inf, 76.1, 98.1, np.inf], labels = np.arange(3))
        df.DiabetesPedigreeFunction = pd.cut(df.DiabetesPedigreeFunction, [-np.inf, .244, .525, .805, 1.11, np.inf], labels = np.arange(5))
        df.Insulin = pd.cut(df.Insulin, [-np.inf, 75, 150, np.inf], labels = np.arange(3))
        df.Pregnancies = pd.cut(df.Pregnancies, [-np.inf, 2, 6, np.inf], labels = np.arange(3))
        df.Glucose = pd.cut(df.Glucose, [-np.inf, 89, 107, 123, 143, np.inf], labels = np.arange(5))

        df = df.astype(int)
        cardinality = { column: df[column].max() + 1 for column in df.columns }
        target = "Outcome"
        synergies = [("Age", "Pregnancies", -1)]
        monotonicities = [("Age", +1), ("Pregnancies", +1), ("BMI", +1), ("DiabetesPedigreeFunction", +1)]
        return df, cardinality, target, synergies, monotonicities
    elif name == "thyroid":
        path = os.path.join(base, "new-thyroid.data")
        columns = ["Hyperthyroid", "T3_resin", "T4", "T3", "TSH", "TSH_diff"]
        df = pd.read_csv(path, names = columns)
        df = df[df.Hyperthyroid.isin((1,2))]
        df.Hyperthyroid = (df.Hyperthyroid == 2)

        for col in ["T3_resin", "T4", "T3", "TSH", "TSH_diff"]:
            df[col] = pd.cut(df[col], 2, labels=np.arange(2))

        target = "Hyperthyroid"
        synergies = [("T3_resin", "T3", +1), ("TSH", "TSH_diff", +1)]
        monotonicities = [("T3_resin", +1), ("T3", +1), ("TSH", +1), ("TSH_diff", +1), ("T4", +1)]

        df = df.astype(int)
        cardinality = { column: df[column].max() + 1 for column in df.columns }
        return df, cardinality, target, synergies, monotonicities
    elif name == "heart-disease":
        path = os.path.join(base, "processed.cleveland.data")
        columns = ["age", "sex", "cp", "trestbps", "chol", "diabetes", 
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca",
        "thal", "heart_disease"]
        df = pd.read_csv(path, names=columns, na_values="?").dropna()
        df = df[["sex", "age", "trestbps", "chol", "diabetes", "heart_disease"]]
        df["heart_disease"] = (df.heart_disease.astype(int) != 0).astype(int)
        df.chol = df.chol > 240
        df.trestbps = pd.cut(df.trestbps, [0, 120, 140, np.inf], labels = np.arange(3))
        df.age = pd.cut(df.age, [0, 40, 60, np.inf], labels = np.arange(3))
        target = "heart_disease"
        synergies = [("sex", "age", -1), ("trestbps", "diabetes", -1 ), ("chol", "diabetes", -1)]
        monotonicities = [("sex", +1), ("age", +1), ("trestbps", +1), ("chol",+1), ("diabetes",+1)]

        df = df.astype(int)
        cardinality = { column: df[column].max() + 1 for column in df.columns }
        return df, cardinality, target, synergies, monotonicities
    
    elif name == "hepatitis":
        pass
    elif name == "lucas_a":
        path = os.path.join(base, "lucas0_train.csv")
        df = pd.read_csv(path, dtype=int)
        names = df.columns.tolist()
        df = df[["Peer_Pressure", "Anxiety", "Smoking"]]# .head(50) # .sample(n = 100, replace=False, random_state=0)
        target = "Smoking"
        monotonicities = [("Peer_Pressure", +1), ("Anxiety", +1)]
        synergies = []
        cardinality = { column: 2 for column in df.columns }
        return df, cardinality, target, synergies, monotonicities
    
    elif name == "lucas_b":
        path = os.path.join(base, "lucas0_train.csv")
        df = pd.read_csv(path, dtype=int)
        names = df.columns.tolist()
        df = df[["Smoking", "Genetics", "Lung_cancer"]]# .head(50) # .sample(n = 100, replace=False, random_state=0)
        target = "Lung_cancer"
        monotonicities = [("Smoking", +1), ("Genetics", +1)]
        synergies = []
        cardinality = { column: 2 for column in df.columns }
        return df, cardinality, target, synergies, monotonicities
    
    elif name == "lucas_c":
        path = os.path.join(base, "lucas0_train.csv")
        df = pd.read_csv(path, dtype=int)
        names = df.columns.tolist()

        df = df[["Allergy", "Lung_cancer", "Coughing"]]# .head(50) # .sample(n = 100, replace=False, random_state=0)
        target = "Coughing"
        monotonicities = [("Allergy", +1), ("Lung_cancer", +1)]
        synergies = []
        cardinality = { column: 2 for column in df.columns }
        return df, cardinality, target, synergies, monotonicities
    
    elif name == "asia":
        path = os.path.join(bn_path, "asia.bif")
        bn = BIFReader(path).get_model()
        df = bn.simulate(2000, seed=0)
        df = df[["bronc", "either", "dysp"]].astype(int)
        target = "dysp"
        monotonicities = [("bronc", +1), ("either", +1)]
        synergies = []
        cardinality = { column: 2 for column in df.columns }
        return df, cardinality, target, synergies, monotonicities
    
    elif name == "cancer":
        path = os.path.join(bn_path, "cancer.bif")
        bn = BIFReader(path).get_model()
        df = bn.simulate(2000, seed=0)
        df = df[["Pollution", "Smoker", "Cancer"]].astype(int)
        target = "Cancer"
        monotonicities = [("Pollution", +1), ("Smoker", +1)]
        synergies = []
        cardinality = { column: 2 for column in df.columns }
        return df, cardinality, target, synergies, monotonicities
    elif name == "adni":
        data_path = os.path.join("raw-data", "ADNI", "Alz_clean_full_bkp.csv")
        data = pd.read_csv(data_path, index_col = "RID")

        demo_path = os.path.join("raw-data", "ADNI", "demographics.csv")
        demo = pd.read_csv(demo_path, index_col = "RID")

        df = data.join(demo, how = "inner")[["MMSCORE", "AGE", "PTGENDER", "DXCURREN"]]
        df["AGE"] = (df.AGE > 75) 
        df["MMSCORE"] = pd.cut(df.MMSCORE, [0, 18, 23, np.inf], labels = np.arange(3))
        df["PTGENDER"] = (df["PTGENDER"] == 1).astype(int)
        df = df.astype(int)
        target = "DXCURREN"
        monotonicities = [("MMSCORE", -1), ("AGE", +1) , ("PTGENDER", +1)]
        synergies = []
        cardinality = { column: df[column].max()+1 for column in df.columns }
        return df, cardinality, target, synergies, monotonicities
    elif name == "ppd":
        data_path = os.path.join("raw-data", "PPD", "train_1.csv")
        data = pd.read_csv(data_path).dropna()
        data_path = os.path.join("raw-data", "PPD", "test_1.csv")
        data2 = pd.read_csv(data_path).dropna()

        data = pd.concat([data, data2])

        data["age"] = pd.cut(data["age"], [0, 3, 4, 7], labels = np.arange(3))
        data["unplanned"] = data[["unplanned_1", "unplanned_2"]].any(axis = 1).astype(int)
        data["partner_support"] = data[[ "marital_satisfaction_2", "marital_satisfaction_3", "social_support_1", "social_support_2", "social_support_3", "social_support_4"]].all(axis=1).astype(int)
        data["partner_support"] = (1-data["marital_satisfaction_1"])*data["partner_support"].astype(int)
        data["ppd"] = data["class label -PPD"]
        df = data[['life_stress', 'maternity_blues', 'unplanned', 'partner_support', 'ppd']].astype(int)
        target = "ppd"
        monotonicities = [("partner_support", -1), ("life_stress", +1) , ("maternity_blues", +1), ("unplanned", +1)]
        synergies = []
        cardinality = { column: df[column].max()+1 for column in df.columns }
        return df, cardinality, target, synergies, monotonicities
    elif name == "rare":
        data_path = os.path.join("raw-data", "Rare", "rare_disease.CSV")
        data = pd.read_csv(data_path, encoding = "latin1").dropna()
        data["Rare"] = (data["Disease Type"] == 'R') | (data["Disease Type"] == 'B')
        data.drop(['Disease Type'], axis = 1, inplace=True)

        df = data[["Posted in an online discussion specific to your condition",
        "Used a social network to remember or memorialize others who suffered from a certain health condition?",
        "How many specialists did you see in the last two years?",
        " Posted a review online of a hospital", 
        "Watched an online video about health or medical issues",
        "Posted on YouTube or other video sharing site",
        "Rare"]].copy()
        df.columns = ["online_discuss", "memorialize", "specialists", "review_hospital", "video1", "video2", "rare"]
        df.online_discuss = (df.online_discuss == "Y")
        df.memorialize = (df.memorialize == "Y")
        df.review_hospital = (df.review_hospital == "Y")
        df["watch_video"] = (df.video1 == "Y") & (df.video2 == "Y")
        df.drop(["video1", "video2"], axis=1, inplace=True)
        df.specialists.replace({ "0": 1, "1 to 2": 1, "3 to 5": 2, "More than 5": 3}, inplace=True) 
        df.specialists = (df.specialists == 3)
        df = df.astype(int)
        target = "rare"
        monotonicities = [("online_discuss", +1), ("memorialize", 1), ("specialists", +1), ("review_hospital", +1)]
        synergies = []
        cardinality = { column: df[column].max()+1 for column in df.columns }
        return df, cardinality, target, synergies, monotonicities
    elif name == "numom2b":
        data_path = os.path.join("raw-data", "numom2b", "numom2b.csv")
        df = pd.read_csv(data_path)
        print (df.METs.mean())
        target = "GDM"
        monotonicities = [("Age", +1), ("BMI", 1), ("PRS", +1), ("Hist", +1),  ("PCOS", +1), ("METs", -1), ("HiBP", +1)]
        synergies = []
        cardinality = { column: df[column].max()+1 for column in df.columns }
        return df, cardinality, target, synergies, monotonicities

