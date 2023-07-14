import os
import pickle
import sys
import pandas as pd
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier


#parameters
n_estimators=yaml.safe_load(open('params.yaml'))['train']['n_estimators']
max_depth=yaml.safe_load(open('params.yaml'))['train']['max_depth']

#load train data
data_file_name=sys.argv[1]
train_data=pd.read_csv(data_file_name)

# model
model_name=sys.argv[2]

x_train=train_data.drop(columns='Churn')
y_train=train_data['Churn']

# model training 
model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
model.fit(x_train, y_train)
model_dir = yaml.safe_load(open('params.yaml'))['train']["model_dir"]
os.makedirs(model_dir, exist_ok=True)
with open(model_dir+"/model.pkl", "wb") as f:
    pickle.dump(model, f)
