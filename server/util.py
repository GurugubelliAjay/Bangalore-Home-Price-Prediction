import os
import json
import pickle
import numpy as np
import pandas as pd
__data_columns = None
__locations = None
__model = None

def get_estimated_price(location,sqft,bhk,bath):
    x=np.zeros(len(__data_columns))
    x[0]=sqft
    x[1]=bath
    x[2]=bhk
    try:
        loc_idx=__data_columns.index(location.lower())
    except:
        loc_idx=-1

    if loc_idx>=0:
        x[loc_idx]=1

    global __model
    return round(__model.predict([x])[0],2)

def get_location_names():
    return __locations

def load_saved_artifacts():
    print('Loading saved artifacts ...')
    global __data_columns
    global __locations
    global __model

    base_path = os.path.dirname(__file__)
    
    columns_path = os.path.join(base_path, "artifacts/columns.json")
    with open(columns_path, 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    model_path = os.path.join(base_path, "artifacts/banglore_home_prices_model.pickle")
    with open(model_path, 'rb') as f:
        __model = pickle.load(f)

    print('Loaded saved artifacts...')

if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar',1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar',1000, 2, 2))
    print(get_estimated_price('Kalhalli',1000, 2, 2))
