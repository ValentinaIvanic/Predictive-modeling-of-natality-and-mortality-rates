from model_imports import *
from check_score_metrics import *
from xgboost import XGBRegressor

data = Data_extracting.get_worldbankForBirths()

data_births = Data_extracting.get_births()

merged_data = pd.merge(data, data_births, on='Year')

merged_data = merged_data.dropna(axis=0, how='any')

x = merged_data[['Year', 'Net migration',  
                        'Rural population growth (annual %)', 'Population in the largest city (% of urban population)']]
y = merged_data['Births']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=100)

model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
model.fit(x_train, y_train)

predictions = model.predict(x_test)

from check_score_metrics import *
print_scores(y_test, predictions)

