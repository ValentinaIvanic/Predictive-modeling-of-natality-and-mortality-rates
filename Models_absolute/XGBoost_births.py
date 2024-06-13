from model_imports import *
from check_score_metrics import *
from xgboost import XGBRegressor, plot_importance
import matplotlib.pyplot as plt


data = Data_extracting.get_worldbankForBirths()
data_births = Data_extracting.get_births()
data_gdp = Data_extracting.get_maddisonProjectData()

merged_data = pd.merge(data, data_births, on='Year')
merged_data = pd.merge(merged_data, data_gdp, on='Year')
merged_data = merged_data.dropna(axis=0, how='any')

# 'Population ages 65 and above (% of total population)' zanemarivo pogoršava   'Rural population growth (annual %)', također mrvicu gore, 'GDP_per_capita_2011_prices' također itd....
#  'Age dependency ratio, young', 'Urban population (% of total population)', 'Rural population (% of total population)',   ostaje isto
# ak zamijenimo Year sa 'Rural population (% of total population)' ostaje isto, mzd i sa ovim ostalim u tom redu
# od ovih kaj su ostale, koja se za koju pogorsava ak se makne

# 'Urban population' pogoršava iako na SMALLDATA poboljšava

x = merged_data[['Rural population (% of total population)', 'Population ages 15-64 (% of total population)', 
                 'Population ages 20-24, female (% of female population)', 
                 'Population growth (annual %)'
                ]]


# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(merged_data[['Rural population (% of total population)', 'Population ages 15-64 (% of total population)', 
#                  'Population ages 20-24, female (% of female population)',
#                  'Population growth (annual %)'
#                 ]])

# scaled_df = pd.DataFrame(scaled_features, columns=['Rural population (% of total population)', 'Population ages 15-64 (% of total population)', 
#                  'Population ages 20-24, female (% of female population)',
#                  'Population growth (annual %)'
#                 ])


# x = scaled_df.values
y = merged_data['Births']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=False, random_state=100)


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



