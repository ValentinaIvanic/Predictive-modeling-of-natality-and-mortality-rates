from model_imports import *
from check_score_metrics import *
from xgboost import XGBRegressor, plot_importance
import matplotlib.pyplot as plt


data = Data_extracting.get_worldbankForBirths()
data_births = Data_extracting.get_births()
data_gdp = Data_extracting.get_maddisonProjectData()
data_economics = Data_extracting.get_worldbankEconomics()
data_inflation = Data_extracting.get_inflation() # | Inflation Rate (%) | Inflation_Annual Change , kvari

merged_data = pd.merge(data, data_births, on='Year')
merged_data = pd.merge(merged_data, data_gdp, on='Year')
merged_data = pd.merge(merged_data, data_economics, on='Year')
merged_data = pd.merge(merged_data, data_inflation, on='Year')

merged_data = merged_data[merged_data['Year'].astype(int) > 1986]
merged_data = merged_data.dropna(axis=0, how='any')

# print(merged_data.columns)

# ['Year', 'Population ages 15-64 (% of total population)',
#        'Population ages 65 and above (% of total population)',
#        'Population ages 20-24, female (% of female population)',
#        'Age dependency ratio, old', 'Age dependency ratio, young',
#        'Urban population (% of total population)', 'Urban population',
#        'Rural population', 'Rural population (% of total population)',
#        'Rural population growth (annual %)', 'Population, total',
#        'Population in the largest city (% of urban population)',
#        'Population growth (annual %)', 'Population in largest city',
#        'Net migration', 'Birth rate, crude (per 1,000 people)', 'Births',
#        'GDP_per_capita_2011_prices', 'population', 'CPI Price, seasonal',
#        'CPI Price',
#        'Exchange rate',
#        'Inflation Rate (%)', 'Inflation_Annual Change']

# rural annual isto, 'Population in the largest city (% of urban population)' isto, bilokoji CPI ostaje isto, 
# bdp malo pogoršava, 'Exchange rate' mrvicu pogoršava, rate inflacije gore,
# 'Inflation_Annual Change' poboljava jeeeej
# year isto
x = merged_data[['Rural population (% of total population)', 'Population ages 15-64 (% of total population)', 
                 'Population ages 20-24, female (% of female population)', 'Urban population',
                 'Population growth (annual %)', 'Inflation_Annual Change'
                ]]
y = merged_data['Births']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, shuffle=False, random_state=100)


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

test_pred = model.predict(x_test)
train_pred = model.predict(x_train)
print_scores(y_test, test_pred)
print_scores(y_train, train_pred)

draw_Graphs.train_test_testPred(merged_data['Year'], y_train, y_test, test_pred, len(y_train))
draw_Graphs.train_test_trainpred_testPred(merged_data['Year'], y_train, y_test, test_pred, train_pred, len(y_train))

