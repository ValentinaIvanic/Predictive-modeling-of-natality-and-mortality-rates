import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from Data_preprocessing import Data_extracting
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)  # Prikaži sve retke
pd.set_option('display.max_columns', None)  # Prikaži sve stupce

#---------------------------------------------------- << >> -----------------------------------------------------

data_original = Data_extracting.get_births()
data_predicted = Data_extracting.get_predictedBirths()


data_combined = pd.concat([data_original, data_predicted], ignore_index=True)

plt.figure(figsize=(12, 6))
plt.plot(data_original['Year'], data_original['Births'], label='Stvaran broj (1960-2022)', color ='green', marker='o')
plt.plot(data_predicted['Year'], data_predicted['Births'], label='Predikcija (2023-2073)', color ='red', marker='*')

plt.xlabel('Godina')
plt.ylabel('Broj rođenih')
plt.title('Predikcija broja rođenih')
plt.legend()
plt.grid(True)
plt.show()
