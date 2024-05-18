import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from Data_preprocessing import Data_extracting

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', None)  # Prikaži sve retke
pd.set_option('display.max_columns', None)  # Prikaži sve stupce

#---------------------------------------------------- << >> -----------------------------------------------------


data_population = Data_extracting.get_population()

# print(data.to_string(index=False))
# print(data.shape)
# print(data.info())

# print(data.isnull().sum())

sns.set_theme()
graph = sns.relplot(data=data_population[['Year','Population']], x='Year', y='Population', kind='line', height=10, linewidth=5, color='red')
graph.set_axis_labels('Godina', 'Stanovništvo(u milijunima)')
plt.title('Stanovništvo RH kroz godine')
plt.show()