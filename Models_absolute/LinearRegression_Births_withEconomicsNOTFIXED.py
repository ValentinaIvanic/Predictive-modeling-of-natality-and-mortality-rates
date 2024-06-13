from model_imports import *
from sklearn.linear_model import Lasso


data = Data_extracting.get_births()
data_features = Data_extracting.get_worldbankForBirths()
data_economics = Data_extracting.get_worldbankEconomics()

merged_data = pd.merge(data, data_features, on='Year')
merged_data = pd.merge(merged_data, data_economics, on='Year')

merged_data = merged_data.dropna(axis=0, how='any')

print(data_economics.columns)
#---------------------------------------------------<< Provjera kolinearnosti >>-----------------------------------------------

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

x_provjera = merged_data[['Year', 'Net migration',  
                        'Rural population growth (annual %)', 'Population in the largest city (% of urban population)']]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_provjera_scaled = scaler.fit_transform(x_provjera)

x_provjera_scaled = sm.add_constant(x_provjera_scaled)
x_provjera_scaled = pd.DataFrame(x_provjera_scaled, columns=['const'] + x_provjera.columns.tolist())

vif = pd.DataFrame()
vif["Stupci"] = x_provjera_scaled.columns
vif["VIF"] = [variance_inflation_factor(x_provjera_scaled, i) for i in range(x_provjera_scaled.shape[1])]


print("\n------------------------------")
print(vif)
print("------------------------------\n")

#----------------------------------------------------<< Model >>-------------------------------------------------------

merged_data = merged_data[merged_data['Year'].astype(int) > 1986]
x = merged_data[['Year', 'Net migration', 'Population in the largest city (% of urban population)',  
                        'Rural population growth (annual %)',  'CPI Price, seasonal', 'Exchange rate']].values
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

from check_score_metrics import *
print_scores(y_test, predicted_data)