import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from Data_preprocessing import Data_extracting

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

data_population = Data_extracting.get_population()
data_births = Data_extracting.get_births()
data_deaths = Data_extracting.get_deaths()

merged_data = pd.merge(data_population, data_births, on='Year')
merged_data = pd.merge(merged_data, data_deaths, on='Year')

print(merged_data.head())

# x = data_population['Year'].values.reshape(-1, 1) #pretvorba iz DataFrame-a u 2d nizove
# y = data_population['Population'].values.reshape(-1, 1)

x = merged_data[['Year', 'Births', 'Deaths']].values  
y = merged_data['Population'].values.reshape(-1, 1)  


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.63, random_state=100)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("R2 score:", r2)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Percentage Error (MAPE):", mape)

