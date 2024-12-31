import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Pre_processing:
    def __init__(self) -> None:
        pass
    
    def delete_col(self,
                   df):
        df = df.drop(['ID',
                      'Delivery_person_ID',
                      'Delivery_person_Age', 
                      'Delivery_person_Ratings',
                      'Order_Date',
                      'Time_Orderd',
                      'Time_Order_picked',
                      'Vehicle_condition',
                      'multiple_deliveries',
                      'Festival'],
                      axis = 1)
        return df
    
    # Finding missing values:
    def rawing(self,
               df):            
        df_raw = df.replace('NaN',
                                    float(np.nan),
                                    regex = True)
        return df_raw

    def label_encoding(self,
                       df_cleant):
        le = LabelEncoder()
        # List of categorical columns to encode
        cat = ['Weatherconditions',
               'Road_traffic_density',
               'Type_of_order',
               'Type_of_vehicle',
               'City']
        # Dictionary to store the label encoders for each column
        encoders = {}
        # Apply label encoding to each categorical column
        for col in cat:
            df_cleant[col] = le.fit_transform(df_cleant[col])
            encoders[col] = le
        # Return the transformed dataframe and the dictionary of label encoders
        return df_cleant, encoders