import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from Data_preprocessing import Data_extracting

import warnings
warnings.filterwarnings('ignore')

#------------------------------------<< Features: year only >>-----------------------------------------
print()
print("Lasso Regression BASIC ----------------------------")
print()

data = Data_extracting.get_births()

x = data['Year'].values.reshape(-1, 1) #pretvorba iz DataFrame-a u 2d nizove
y = data['Births'].values.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.75, random_state=100)

model = Lasso(alpha=0.1)
model.fit(x_train, y_train)

predictions = model.predict(x_test)

from check_score_metrics import *

print_scores(y_test, predictions)


#------------------------------------<< Features: columns_birth >>-----------------------------------------
    # columns_births = ['Year', 'Net migration', 'Population in largest city', 'Population growth (annual %)', 'Population, total', 
    #                   'Rural population', 'Rural population (% of total population)', 'Rural population growth (annual %)', 
    #                   'Urban population (% of total population)', 'Urban population']
print()
print("Lasso Regression BIG MODEL ----------------------------")
print()

data_features = Data_extracting.get_worldbankForBirths()

merged_data = pd.merge(data, data_features, on='Year')

merged_data = merged_data.dropna(axis=0, how='any')

merged_data[['Year', 'Urban population (% of total population)',
       'Urban population', 'Rural population',
       'Rural population (% of total population)',
       'Rural population growth (annual %)', 'Population, total',   
       'Population growth (annual %)', 'Population in largest city',
       'Net migration']] = merged_data[['Year', 'Urban population (% of total population)',
                                        'Urban population', 'Rural population',
                                        'Rural population (% of total population)',
                                        'Rural population growth (annual %)', 'Population, total',   
                                        'Population growth (annual %)', 'Population in largest city',
                                        'Net migration']].astype(float)

x = merged_data[['Year',
       'Urban population', 'Rural population',
       'Population, total',
       'Net migration']].values
y = merged_data[['Births']].values.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=50)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

model_big = Lasso(alpha=0.1)
model_big.fit(x_train_scaled, y_train)

print(model_big.coef_)

x_test_scaled = scaler.fit_transform(x_test)
predicted_data = model_big.predict(x_test_scaled)

print_scores(y_test, predicted_data)