import pandas as pd
import numpy as np
from geopy.distance import geodesic
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt
import plotly.express as px
import folium
import openrouteservice
from openrouteservice import convert
from openrouteservice import convert
import json
import sys
from pathlib import Path

st.set_page_config(page_title = "Food Deliveri Dashboard",
                   layout = "wide",
                   initial_sidebar_state = "expanded")
alt.themes.enable("dark")

# Main Streamlit app
st.title("Food Delivery Dashboard")

## ---------------- Tab 1 ----------------
col1, col2, col3 = st.columns([1, 1, 3])

def initialize_state():
    if 'user_data' not in st.session_state:
        st.session_state.sample = {}

initialize_state()

with col1:
    r_latitude = st.number_input(label = 'Please enter restaurant latitude:',
                                step = 0.000000001,
                                format = '%f')
    r_longitude = st.number_input(label = 'Please enter restaurant longitude:',
                                step = 0.000000001,
                                format = '%f')
    dl_latitude = st.number_input(label = 'Please enter delivery location latitude:',
                                step = 0.000000001,
                                format = '%f')
    dl_longitude = st.number_input(label = 'Please enter delivery location longitude:',
                                step = 0.000000001,
                                format = '%f')
    weathercondition = st.selectbox('Please select weather condition:', 
                                        ('Fog', 'Stormy', 'Cloudy', 'Sandstorms', 'Windy', 'Sunny'))

with col2:
    trafficDensity = st.selectbox('Please select traffic density:', 
                                        ('Low', 'Jam', 'Medium', 'High'))
    typeOfOrder = st.selectbox('Please select type of order:',
                                    ('Snack', 'Meal', 'Drinks', 'Buffet'))
    vehicle = st.selectbox('Please select type of vehicle',
                                        ('motorcycle', 'scooter', 'electric_scooter', 'bicycle'))
    city = st.selectbox('Please select deliveri condition:',
                                        ('Metropolitian', 'Urban', 'Semi-Urban', 'NaN'))
    

if st.button('Save!'):
    # information storing:
    st.session_state.sample['Restaurant_latitude'] = r_latitude,
    st.session_state.sample['Restaurant_longitude'] = r_longitude,
    st.session_state.sample['Delivery_location_latitude'] = dl_latitude,
    st.session_state.sample['Delivery_location_longitude'] = dl_longitude,
    st.session_state.sample['Weatherconditions'] = weathercondition,
    st.session_state.sample['Road_traffic_density'] = trafficDensity,
    st.session_state.sample['Type_of_order'] = typeOfOrder,
    st.session_state.sample['Type_of_vehicle'] = vehicle,
    st.session_state.sample['City'] = city
    st.write('Saved!')


sample = st.session_state.sample

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
# data_loader = Data_loader()
df = pd.DataFrame(st.session_state.sample)

from data_preprocessing.preprocessor import Pre_processing
preprocessing = Pre_processing()
df_raw = preprocessing.rawing(df)

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
                'Delivery_location_longitude']

df_float = df_float.drop(['distance'],
                         axis=1)

df_cleant = transformer.filling(df_float,
                                num_attribs)

df_cleant, encoders = preprocessing.label_encoding(df_cleant)

import joblib
config = joblib.load('./dtree_cf.joblib')

from model.training import dtree_predictor
dt_pred = dtree_predictor(df_cleant, config)
print(dt_pred)

df_map =  pd.DataFrame({'latitude_col': [r_latitude,
                                        dl_latitude],
                    'longitude_col': [r_longitude,
                                        dl_longitude],
                    'color_col': ['#CB0B0B',
                                    '#3468C1']})

## Example content in second column of Tab 1
with col3:
    # st.scatter_chart(strm.leanear_r_scatter)
    ################ MAP #################
    st.map(df_map,
            latitude = 'latitude_col',
            longitude = 'longitude_col',
            color = 'color_col')
    st.write(map)
## Prediction Deliveri Time:
st.markdown(f'#### Prediction time is: {dt_pred} minutes.')