import pandas as pd
import numpy as np
from geopy.distance import geodesic
from datetime import datetime, timedelta
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from pathlib import Path
import sys

prd_columns = ['Restaurant_latitude',
               'Restaurant_longitude',
               'Delivery_location_latitude',
               'Delivery_location_longitude',
               'Weatherconditions',
               'Road_traffic_density',
               'Type_of_order',
               'Type_of_vehicle',
               'City']

cat_attribs = ['Weatherconditions',
               'Road_traffic_density', 
               'Type_of_order',
               'Type_of_vehicle',
               'City']

cols = ['Restaurant_latitude', 
        'Restaurant_longitude', 
        'Delivery_location_latitude', 
        'Delivery_location_longitude']


class Data_loader:
    def __init__(self) -> None:
        pass

    #  Loading Data:
    def loading(self, data_path):
        df = pd.read_csv(data_path)
        return df


class Data_cleaning:
    def __init__(self) -> None:
        pass

  # Data cleaning:
    def spam_spling(self, df_raw):
        # Removing the string part from Weatherconditions & time taken:
        df_raw['Weatherconditions'] = df_raw['Weatherconditions'].str.split(" ", expand = True)[1]
        if 'Time_taken(min)' in df_raw.columns:
            df_raw['Time_taken(min)'] = df_raw['Time_taken(min)'].str.split(" ", expand = True)[1]
        return df_raw
 
       
class Converting:
    def __init__(self) -> None:
        pass
     
    def to_float(self, df_raw):
        df_raw['Time_taken(min)'] = df_raw['Time_taken(min)'].astype('float64')
        return df_raw
    
    def abs_location(self, df_float):
        for col in cols:
            df_float[col] = abs(df_float[col])
        return df_float
    
    def loc_cordinate(self,
                     df_float,
                     restaurant_loc_df,
                     delivery_loc_df):
        df_float['distance'] = np.zeros(len(df_float))
        for i in range(len(df_float)):
            df_float['distance'].loc[i] = geodesic(restaurant_loc_df[i], delivery_loc_df[i])
        df_float['distance'] = df_float['distance'].astype('str').str.extract('(\d+)')
        df_float['distance'] = df_float['distance'].astype('float64')
        return df_float['distance']
    
    
class Transformer_:
    def __init__(self) -> None:
        pass

    def to_dist(self, df_float):
         restaurant_cordinates_df = df_float[['Restaurant_latitude', 'Restaurant_longitude']].to_numpy()
         delivery_location_cordinates_df = df_float[['Delivery_location_latitude', 'Delivery_location_longitude']].to_numpy()
         df_float['distance'] = Converting.loc_cordinate(self,
                                                         df_float,
                                                         restaurant_cordinates_df,
                                                         delivery_location_cordinates_df)
         return df_float
    
    def filling(self,
                df_float,
                num_attribs):
        numerical_transformer = SimpleImputer(strategy = 'median')
        categorical_transformer = SimpleImputer(strategy = 'most_frequent')
        preprocessor = ColumnTransformer(transformers = [('num',
                                                          numerical_transformer,
                                                          num_attribs),
                                                          ('cat',
                                                           categorical_transformer,
                                                           cat_attribs)])
        df_cleant = pd.DataFrame(preprocessor.fit_transform(df_float),
                                 columns = num_attribs + cat_attribs)
        return df_cleant

    def df_transformed(self, df_cleant):
        # The numerical features:
        df_num = pd.DataFrame(df_cleant[['Delivery_person_Age',
                                         'Delivery_person_Ratings',
                                         'Vehicle_condition',
                                         'distance',
                                         'Time_taken(min)',
                                         'order_preparation_time',
                                         'multiple_deliveries']])
        df_transformed = pd.concat([df_num, df_cleant], axis = 1)
        return df_transformed