# Importing pakages:
import numpy as np
import pandas as pd
from pathlib import Path
import sys

current_dir = Path(__file__).resolve().parent
preprocessing_folder = (current_dir / r'./data_preprocessing').resolve()

sys.path.append(str(preprocessing_folder))
current_dir = Path(__file__).resolve().parent
model_folder = (current_dir / r'./model').resolve()

sys.path.append(str(model_folder))


# Data loading:
from data_preprocessing.data_loader import Data_loader
data_loader = Data_loader()
df = data_loader.loading(r'./test.csv')

from data_preprocessing.preprocessor import Pre_processing
preprocessing = Pre_processing()
df = preprocessing.delete_col(df)
df_raw = preprocessing.rawing(df)

from data_preprocessing.data_loader import Data_cleaning
data_cleaning = Data_cleaning()
df_raw = data_cleaning.spam_spling(df_raw)

from data_preprocessing.data_loader import Converting
converting = Converting()
cols = ['Restaurant_latitude', 
        'Restaurant_longitude', 
        'Delivery_location_latitude', 
        'Delivery_location_longitude']  

df_float = converting.abs_location(df_raw)

from data_preprocessing.data_loader import Transformer_
transformer = Transformer_()
df_float = transformer.to_dist(df_float)

num_attribs = ['Restaurant_latitude', 
                'Restaurant_longitude', 
                'Delivery_location_latitude', 
                'Delivery_location_longitude',
                'distance']
df_cleant = transformer.filling(df_float,
                                num_attribs)

df_cleant, encoders = preprocessing.label_encoding(df_cleant)

import joblib
config = joblib.load('./dtree_cf.joblib')
X = df_cleant.drop(['distance'],
                   axis = 1)

x = X.iloc[4].to_dict()
from model.training import dtree_predictor
dt_pred = dtree_predictor(x, config)
print(dt_pred)



# for i in range(X):
#     if (sample.keys == X.columns) and (sample.values == X.values):
#         x = X.iloc[i]
#     return i
# value_list = []
# for i in sample.values():
#     value_list.append()

# for i in value_list:
#     if value_list == X.iloc[i].to_list():
#         x = X.iloc[i].to_dict()