from model_imports import *
from check_score_metrics import *
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler

data = Data_extracting.get_worldbankForDeaths()
data_deaths = Data_extracting.get_deaths()
data_gdp = Data_extracting.get_maddisonProjectData()

data_economics = Data_extracting.get_worldbankEconomics() 
data_inflation = Data_extracting.get_inflation() 

merged_data = pd.merge(data, data_deaths, on='Year')
merged_data = pd.merge(merged_data, data_gdp, on='Year')
merged_data = pd.merge(merged_data, data_economics, on='Year')
merged_data = pd.merge(merged_data, data_inflation, on='Year')

merged_data = merged_data[merged_data['Year'].astype(int) > 1986]
merged_data = merged_data.dropna(axis=0, how='any')

scaler = StandardScaler()
scaled_features = scaler.fit_transform(merged_data[['Year', 
                'Life expectancy at birth, total (years)', 
                'Age dependency ratio, old', 'Exchange rate', 
                'CPI Price, seasonal', 'CPI Price']])

scaled_df = pd.DataFrame(scaled_features, columns=['Year', 
                'Life expectancy at birth, total (years)', 
                'Age dependency ratio, old', 'Exchange rate', 
                'CPI Price, seasonal', 'CPI Price'])

x = scaled_df.values
y = merged_data['Deaths']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, shuffle = False, random_state=100)


print("###########################################################################")

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
