from model_imports import *
from check_score_metrics import *
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler

data = Data_extracting.get_worldbankForBirths()
data_births = Data_extracting.get_births()
data_gdp = Data_extracting.get_maddisonProjectData()
data_economics = Data_extracting.get_worldbankEconomics()
data_inflation = Data_extracting.get_inflation() 

merged_data = pd.merge(data, data_births, on='Year')
merged_data = pd.merge(merged_data, data_gdp, on='Year')
merged_data = pd.merge(merged_data, data_economics, on='Year')
merged_data = pd.merge(merged_data, data_inflation, on='Year')

merged_data = merged_data[merged_data['Year'].astype(int) > 1986]
merged_data = merged_data.dropna(axis=0, how='any')

scaler = StandardScaler()
scaled_features = scaler.fit_transform(merged_data[['Year',
                        'Rural population growth (annual %)', 
                        'Population in the largest city (% of urban population)', 
                        ]])

scaled_df = pd.DataFrame(scaled_features, columns=['Year',
                        'Rural population growth (annual %)', 
                        'Population in the largest city (% of urban population)', 
                        ])


x = scaled_df.values
y = merged_data['Births']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, random_state=100, shuffle=False)


print("###########################################################################")
model1 = SARIMAX(y_train, exog = x_train, order=(20, 1, 3), seasonal_order=(0, 0, 0, 0)).fit()
predictions = model1.forecast(steps = len(y_test), exog = x_test)

test_pred = model1.forecast(steps = len(y_test), exog = x_test)
train_pred = model1.forecast(steps = len(y_train), exog = x_train)
print_scores(y_test, test_pred)
print_scores(y_train, train_pred)

draw_Graphs.train_test_testPred(merged_data['Year'], y_train, y_test, test_pred, len(y_train))
draw_Graphs.train_test_trainpred_testPred(merged_data['Year'], y_train, y_test, test_pred, train_pred, len(y_train))