import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from Data_preprocessing import Data_extracting

#---------------------------------------------<< >>--------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def get_data():

    data_births = Data_extracting.get_births()

    data_demFeatures = Data_extracting.get_worldbankForBirths()
    data_ecoFeatures = Data_extracting.get_worldbankEconomics()
    data_gdp = Data_extracting.get_maddisonProjectData()
    data_manufacturing = Data_extracting.get_manufacturingOutput()
    data_inflation = Data_extracting.get_inflation()
    data_employment = Data_extracting.get_unemployment()
    data_trade = Data_extracting.get_Imports_Exports()


    merged_data = pd.merge(data_births, data_demFeatures, on='Year')
    merged_data = pd.merge(merged_data, data_ecoFeatures, on='Year')
    merged_data = pd.merge(merged_data, data_gdp, on='Year')
    merged_data = pd.merge(merged_data, data_manufacturing, on='Year')
    merged_data = pd.merge(merged_data, data_inflation, on='Year')
    merged_data = pd.merge(merged_data, data_employment, on='Year')
    merged_data = pd.merge(merged_data, data_trade, on='Year')

    merged_data = merged_data.drop(columns=['Population, total', 'population', 
                                            'Birth rate, crude (per 1,000 people)', 
                                            'Population in largest city',
                                            'Rural population', 'Urban population'])
    
    return merged_data