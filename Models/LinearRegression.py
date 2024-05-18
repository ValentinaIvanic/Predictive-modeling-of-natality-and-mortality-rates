import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from Data_preprocessing import Data_extracting

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_rows', None)  # Prikaži sve retke
pd.set_option('display.max_columns', None)  # Prikaži sve stupce

data_population = Data_extracting.get_population()

# print(data.to_string(index=False))
# print(data.shape)
# print(data.info())


# print(data.isnull().sum())

sns.set_theme()

# Vaš ostatak koda za crtanje grafa
sns.relplot(data=data_population[['Year','Population']], x='Year', y='Population', kind='line', height=10, linewidth=3, color='purple')
plt.title('Population Size')
plt.show()