from model_imports import *
from check_score_metrics import *
from xgboost import XGBRegressor, plot_importance

data = Data_extracting.get_worldbankForBirths()
data_births = Data_extracting.get_births()

merged_data = pd.merge(data, data_births, on='Year')

# može se dodati Year za jos bolje(r2=0.985), sad ke r2=0.975, to za 'Age dependency ratio, young' kaj je cheating right??
# za samo 'Net migration', 'Population in the largest city (% of urban population)' je r2 = 0.726 HAAAAAAAAA? (ni jedno zasebno ne dela dobro)
x = merged_data[['Net migration',  
                'Population in the largest city (% of urban population)', 'Rural population growth (annual %)'
                ]]
y = merged_data['Birth rate, crude (per 1,000 people)']

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



