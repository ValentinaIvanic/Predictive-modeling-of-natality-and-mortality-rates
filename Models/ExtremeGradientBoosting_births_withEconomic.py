from model_imports import *
from check_score_metrics import *
from xgboost import XGBRegressor, plot_importance

data = Data_extracting.get_worldbankForBirths()
data_births = Data_extracting.get_births()
data_economics = Data_extracting.get_worldbankEconomics()

merged_data = pd.merge(data, data_births, on='Year')
merged_data = pd.merge(merged_data, data_economics, on='Year')
merged_data = merged_data[merged_data['Year'].astype(int) > 1986]
merged_data = merged_data.dropna(axis=0, how='any')


x = merged_data[['Year', 'Net migration',  
                'Rural population growth (annual %)', 
                'Population in the largest city (% of urban population)',
                'Exchange rate, new LCU per USD extended backward, period average,,',
                'CPI Price, seas. adj.,,,',
                'CPI Price,not seas.adj,,,',
                'Age dependency ratio, young',  
                'Population ages 15-64 (% of total population)',
                'Population ages 20-24, female (% of female population)']]
y = merged_data['Births']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=100)


import matplotlib.pyplot as plt
from xgboost import XGBRegressor, DMatrix

eval_set = [(x_train, y_train), (x_test, y_test)]


model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
model.fit(x_train, y_train, eval_metric='mae', eval_set=eval_set, verbose=True)

predictions = model.predict(x_test)
print_scores(y_test, predictions)

fig, ax = plt.subplots()
plot_importance(model, ax=ax)
plt.show()

#----------------------------------------------<< Overfitting checks >>-----------------------------------------

pred_train = model.predict(x_train)
print_scores(y_train, pred_train)


results = model.evals_result()

eophe = len(results['validation_0']['mae'])
x_axis = range(0, eophe)

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['mae'], label='Train')
ax.plot(x_axis, results['validation_1']['mae'], label='Test')
ax.legend()
plt.xlabel('Broj iteracija')
plt.ylabel('MAE')
plt.title('XGB train & test errors')
plt.show()



