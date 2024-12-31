import pandas as pd
import numpy as np


def dtree_predictor(sample, config):
    #model & config loading:
    model = config['model']

    #data transform
    # sample = x.values
    # sample = np.expand_dims(sample, axis=0)
    
    # prediction:
    result = model.predict(sample)
    return result[0]