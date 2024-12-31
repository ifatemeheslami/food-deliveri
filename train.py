# Importing pakages:
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from geopy.distance import geodesic 
import gc
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
preprocessing_folder = (current_dir / r'./data_preprocessing').resolve()
sys.path.append(str(preprocessing_folder))

# Data loading:
from data_preprocessing.data_loader import Data_loader
data_loader = Data_loader()
df = data_loader.loading(r'./train.csv')

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
df_float = converting.to_float(df_raw)

df_float = converting.abs_location(df_float)

from data_preprocessing.data_loader import Transformer_
transformer = Transformer_()
df_float = transformer.to_dist(df_float)

num_attribs = ['Restaurant_latitude', 
                'Restaurant_longitude', 
                'Delivery_location_latitude', 
                'Delivery_location_longitude',
                'distance',
                'Time_taken(min)']

df_cleant = transformer.filling(df_float,
                                num_attribs)


df_cleant, encoders = preprocessing.label_encoding(df_cleant)


# Train/test split:
from sklearn.model_selection import train_test_split
X = df_cleant.drop(['Time_taken(min)', 'distance'],
                   axis = 1)
Y = df_cleant['Time_taken(min)']

x_train, x_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size = 0.2,
                                                    random_state = 13)

from sklearn.tree import DecisionTreeRegressor
dtree = DecisionTreeRegressor(random_state = 13)
dtree.fit(x_train,
          y_train)
pred = dtree.predict(x_test)

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,
                         pred))

import joblib
dtree_config = {'model': dtree}
joblib.dump(dtree_config,
            './dtree_cf.joblib')