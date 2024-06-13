from model_imports import *
from check_score_metrics import *
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler

data = Data_extracting.get_worldbankForDeaths()
data_births = Data_extracting.get_deaths()
data_gdp = Data_extracting.get_maddisonProjectData()

merged_data = pd.merge(data, data_births, on='Year')
merged_data = pd.merge(merged_data, data_gdp, on='Year')
merged_data = merged_data.dropna(axis=0, how='any')


# bez survival to age65 je isto dosta dobrooo
scaler = StandardScaler()
scaled_features = scaler.fit_transform(merged_data[['Year', 
                'Life expectancy at birth, total (years)', 
                'Age dependency ratio, old', 
                'Survival to age 65, male (% of cohort)']])

scaled_df = pd.DataFrame(scaled_features, columns=['Year', 
                'Life expectancy at birth, total (years)', 
                'Age dependency ratio, old', 
                'Survival to age 65, male (% of cohort)'])

x = scaled_df.values
y = merged_data['Deaths']
print(merged_data['Survival to age 65, male (% of cohort)'])
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, shuffle = False, random_state=100)


print("###########################################################################")
# model1 = SARIMAX(y_train, exog = x_train, order=(20, 1, 4)).fit()
# predictions = model1.forecast(steps = len(y_test), exog = x_test)
# test_pred = model1.forecast(steps = len(y_test), exog = x_test)
# train_pred = model1.forecast(steps = len(y_train), exog = x_train)

# print_scores(y_test, test_pred)
# print_scores(y_train, train_pred)

# draw_Graphs.train_test_testPred(merged_data['Year'], y_train, y_test, test_pred, len(y_train))
# draw_Graphs.train_test_trainpred_testPred(merged_data['Year'], y_train, y_test, test_pred, train_pred, len(y_train))

# model2 = SARIMAX(y_train, exog = x_train, order=(1, 1, 4)).fit()
# predictions = model2.forecast(steps = len(y_test), exog = x_test)
# print_scores(y_test, predictions)

model3 = SARIMAX(y_train, exog = x_train, order=(5, 1, 3)).fit()

print(model3)

coefficients = model3.params
p_values = model3.pvalues

summary_df = pd.DataFrame({'Coefficient': coefficients, 'p-value': p_values})
print(summary_df)

test_pred = model3.forecast(steps = len(y_test), exog = x_test)
train_pred = model3.forecast(steps = len(y_train), exog = x_train)

print_scores(y_test, test_pred)
print_scores(y_train, train_pred)

draw_Graphs.train_test_testPred(merged_data['Year'], y_train, y_test, test_pred, len(y_train))
draw_Graphs.train_test_trainpred_testPred(merged_data['Year'], y_train, y_test, test_pred, train_pred, len(y_train))
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