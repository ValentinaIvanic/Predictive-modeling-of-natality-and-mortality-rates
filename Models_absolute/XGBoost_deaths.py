from model_imports import *
from check_score_metrics import *
from xgboost import XGBRegressor, plot_importance
import matplotlib.pyplot as plt


data = Data_extracting.get_worldbankForDeaths()
data_deaths = Data_extracting.get_deaths()
data_gdp = Data_extracting.get_maddisonProjectData()

merged_data = pd.merge(data, data_deaths, on='Year')
merged_data = pd.merge(merged_data, data_gdp, on='Year')
merged_data = merged_data.dropna(axis=0, how='any')

# print(merged_data.columns)
# ['Year', 'Population ages 15-64 (% of total population)',
#        'Population ages 65 and above (% of total population)',
#        'Population ages 20-24, female (% of female population)',
#        'Age dependency ratio, old', 'Age dependency ratio, young',
#        'Life expectancy at birth, total (years)',
#        'Urban population (% of total population)', 'Urban population',
#        'Survival to age 65, female (% of cohort)',
#        'Survival to age 65, male (% of cohort)', 'Rural population',
#        'Rural population (% of total population)',
#        'Rural population growth (annual %)', 'Population, total',
#        'Population in the largest city (% of urban population)',
#        'Population growth (annual %)', 'Population in largest city',
#        'Population ages 80 and above, male (% of male population)',
#        'Net migration', 'Death rate, crude (per 1,000 people)', 'Deaths',
#        'GDP_per_capita_2011_prices', 'population']


# 'Rural population' isto, 'Urban population' isto, 'Population in largest city' isto , 'Population ages 65 and above (% of total population)',
# 'Population ages 20-24, female (% of female population)' pogoršava,  'Population ages 15-64 (% of total population)' pogoršava

# male pogoršavaju 'Survival to age 65, male (% of cohort)', 
#                'Rural population (% of total population)', 'Rural population growth (annual %)', 

# zanemarivo pogoršava 'GDP_per_capita_2011_prices' , 'Survival to age 65, female (% of cohort)', 'Urban population (% of total population)', 


x = merged_data[['Life expectancy at birth, total (years)', 
                'Net migration', 'Population growth (annual %)', 
                'Population ages 80 and above, male (% of male population)', 
                'Population in the largest city (% of urban population)',
                'Age dependency ratio, old'
                ]]
y = merged_data['Deaths']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=False, random_state=100)
eval_set = [(x_train, y_train), (x_test, y_test)]


model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
model.fit(x_train, y_train, eval_metric='mae', eval_set=eval_set, verbose=True)

predictions = model.predict(x_test)
print_scores(y_test, predictions)

fig, ax = plt.subplots()
plot_importance(model, ax=ax)
ax.set_xlabel('F score', fontsize = 18)
ax.set_ylabel('Varijable', fontsize = 18)
ax.set_title('Važnost varijabli u modelu', fontsize = 18)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.show()

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

test_pred = model.predict(x_test)
train_pred = model.predict(x_train)
print_scores(y_test, test_pred)
print_scores(y_train, train_pred)

draw_Graphs.train_test_testPred(merged_data['Year'], y_train, y_test, test_pred, len(y_train))
draw_Graphs.train_test_trainpred_testPred(merged_data['Year'], y_train, y_test, test_pred, train_pred, len(y_train))