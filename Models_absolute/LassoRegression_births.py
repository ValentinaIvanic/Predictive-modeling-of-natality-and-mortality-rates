from model_imports import *
from sklearn.linear_model import Lasso


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


x = merged_data[['Year', 'Net migration', 'Population in the largest city (% of urban population)',  
                        'Rural population growth (annual %)']].values
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