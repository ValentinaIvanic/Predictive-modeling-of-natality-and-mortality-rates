from model_imports import *
from check_score_metrics import *
from statsmodels.tsa.statespace.sarimax import SARIMAX

data = Data_extracting.get_worldbankForDeaths()
data_births = Data_extracting.get_deaths()

merged_data = pd.merge(data, data_births, on='Year')
merged_data = merged_data.dropna(axis=0, how='any')

x = merged_data[['Year', 'Survival to age 65, male (% of cohort)',
                 'Rural population (% of total population)',
                 'Net migration', 'Population growth (annual %)' ]]
y = merged_data['Deaths']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=100)


print("###########################################################################")
# model1 = SARIMAX(y_train, exog = x_train, order=(20, 1, 4)).fit()
# predictions = model1.forecast(steps = len(y_test), exog = x_test)
# print_scores(y_test, predictions)

# model2 = SARIMAX(y_train, exog = x_train, order=(1, 1, 4)).fit()
# predictions = model2.forecast(steps = len(y_test), exog = x_test)
# print_scores(y_test, predictions)

model3 = SARIMAX(y_train, exog = x_train, order=(5, 1, 3)).fit()
predictions = model3.forecast(steps = len(y_test), exog = x_test)
print_scores(y_test, predictions)

# model4 = SARIMAX(y_train, exog = x_train, order=(10, 1, 5)).fit()
# predictions = model4.forecast(steps = len(y_test), exog = x_test)
# print_scores(y_test, predictions)

# model5 = SARIMAX(y_train, exog = x_train, order=(20, 1, 10)).fit()
# predictions = model5.forecast(steps = len(y_test), exog = x_test)
# print_scores(y_test, predictions)

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, random_state=100)

# model6 = SARIMAX(y_train, exog = x_train, order=(15, 1, 5)).fit()
# predictions = model6.forecast(steps = len(y_test), exog = x_test)
# print_scores(y_test, predictions)

# model7 = SARIMAX(y_train, exog = x_train, order=(3, 1, 5)).fit()
# predictions = model7.forecast(steps = len(y_test), exog = x_test)
# print_scores(y_test, predictions)