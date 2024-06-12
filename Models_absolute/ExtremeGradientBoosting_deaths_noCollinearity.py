from model_imports import *
from check_score_metrics import *
from xgboost import XGBRegressor

data = Data_extracting.get_worldbankForDeaths()

data_deaths = Data_extracting.get_deaths()

merged_data = pd.merge(data, data_deaths, on='Year')

merged_data = merged_data.dropna(axis=0, how='any')

x = merged_data[['Year', 'Survival to age 65, male (% of cohort)',
                 'Rural population (% of total population)',
                 'Net migration', 'Population growth (annual %)' ]]
y = merged_data['Deaths']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=100)


import matplotlib.pyplot as plt
from xgboost import XGBRegressor

eval_set = [(x_train, y_train), (x_test, y_test)]


model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth = 5)
model.fit(x_train, y_train, eval_metric='mae', eval_set=eval_set, verbose=True)

predictions = model.predict(x_test)
print_scores(y_test, predictions)

#----------------------------------------------<< Overfitting checks >>-----------------------------------------

pred_train = model.predict(x_train)
print_scores(y_train, pred_train)


results = model.evals_result()

epochs = len(results['validation_0']['mae'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['mae'], label='Train')
ax.plot(x_axis, results['validation_1']['mae'], label='Test')
ax.legend()
plt.xlabel('Number of Iterations')
plt.ylabel('Mean Absolute Error')
plt.title('XGBoost Training and Validation Error')
plt.show()