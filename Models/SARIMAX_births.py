from model_imports import *
from check_score_metrics import *
from statsmodels.tsa.statespace.sarimax import SARIMAX

data = Data_extracting.get_worldbankForBirths()
data_births = Data_extracting.get_births()
data_gdp = Data_extracting.get_maddisonProjectData()


merged_data = pd.merge(data, data_births, on='Year')
merged_data = pd.merge(merged_data, data_gdp, on='Year')
merged_data = merged_data.dropna(axis=0, how='any')

x = merged_data[['Year', 'Net migration',  
                        'Rural population growth (annual %)', 'Population in the largest city (% of urban population)']]
y = merged_data['Births']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, random_state=100)

# order(_, 1, 1): prvi broj najbolje 1-3 (r2 oko 0.759), te od 7 (r2 oko 0.759) raste sve do 20 (r2 oko 0.828), izmedu 3 i 7 je r2 oko 0.73
# order(20, _, 1): samo sa 1 radi

print("###########################################################################")
model1 = SARIMAX(y_train, exog = x_train, order=(20, 1, 4), seasonal_order=(0, 0, 0, 0)).fit()
predictions = model1.forecast(steps = len(y_test), exog = x_test)
print_scores(y_test, predictions)

# model2 = SARIMAX(y_train, exog = x_train, order=(1, 1, 1)).fit()
# predictions = model2.forecast(steps = len(y_test), exog = x_test)
# print_scores(y_test, predictions)

# model3 = SARIMAX(y_train, exog = x_train, order=(5, 1, 2)).fit()
# predictions = model3.forecast(steps = len(y_test), exog = x_test)
# print_scores(y_test, predictions)

# model4 = SARIMAX(y_train, exog = x_train, order=(10, 1, 1)).fit()
# predictions = model4.forecast(steps = len(y_test), exog = x_test)
# print_scores(y_test, predictions)

# model5 = SARIMAX(y_train, exog = x_train, order=(20, 1, 2)).fit()
# predictions = model5.forecast(steps = len(y_test), exog = x_test)
# print_scores(y_test, predictions)

# model6 = SARIMAX(y_train, exog = x_train, order=(15, 1, 2)).fit()
# predictions = model6.forecast(steps = len(y_test), exog = x_test)
# print_scores(y_test, predictions)

# model7 = SARIMAX(y_train, exog = x_train, order=(3, 1, 1)).fit()
# predictions = model7.forecast(steps = len(y_test), exog = x_test)
# print_scores(y_test, predictions)
