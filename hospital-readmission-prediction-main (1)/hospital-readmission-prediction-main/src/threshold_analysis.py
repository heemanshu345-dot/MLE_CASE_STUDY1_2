import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT,"data","diabetic_data.csv")).replace("?",np.nan)
df["target_30day"]=(df["readmitted"]=="<30").astype(int)
drop=["encounter_id","patient_nbr","readmitted","target_30day","weight","payer_code","medical_specialty"]
X=df.drop(columns=drop,errors="ignore"); y=df["target_30day"]
cats=[c for c in X.columns if X[c].dtype=="object"]; nums=[c for c in X.columns if c not in cats]
pre=ColumnTransformer([
("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),nums),
("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cats)])
pipe=Pipeline([("pre",pre),("model",LogisticRegression(penalty="l2",C=1.0,solver="liblinear",max_iter=2000,class_weight="balanced"))])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
pipe.fit(Xtr,ytr); p=pipe.predict_proba(Xte)[:,1]

rows=[]
for t in [0.20,0.30,0.40,0.50,0.60,0.70,0.80]:
    pred=(p>=t).astype(int); tn,fp,fn,tp=confusion_matrix(yte,pred).ravel()
    rows.append([t,tn,fp,fn,tp,precision_score(yte,pred,zero_division=0),recall_score(yte,pred,zero_division=0)])
out=pd.DataFrame(rows,columns=["threshold","TN","FP","FN","TP","precision","recall"])
out.to_csv(os.path.join(ROOT,"results","threshold_analysis.csv"),index=False)
print(out.to_string(index=False))
