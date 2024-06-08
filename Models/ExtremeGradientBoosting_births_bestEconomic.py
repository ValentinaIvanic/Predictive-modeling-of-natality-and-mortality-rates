from model_imports import *
from check_score_metrics import *
from xgboost import XGBRegressor, plot_importance

data = Data_extracting.get_worldbankForBirths()
data_births = Data_extracting.get_births()
data_economics = Data_extracting.get_worldbankEconomics()
data_manufacturing = Data_extracting.get_manufacturingOutput()
data_inflation = Data_extracting.get_inflation()
data_employment = Data_extracting.get_unemployment()
data_trade = Data_extracting.get_Imports_Exports()
data_firstMarriageRatio = Data_extracting.get_ratioFirstTimeMarriedToPopulation()
data_HICP = Data_extracting.get_HICP_Eurostat()
data_consumerIndexes = Data_extracting.get_ConsumerIndexes()
data_salaryIndexes = Data_extracting.get_AverageSalaryByMonth()


merged_data = pd.merge(data, data_births, on='Year')
merged_data = pd.merge(merged_data, data_economics, on='Year')
merged_data = pd.merge(merged_data, data_manufacturing, on='Year')
merged_data = pd.merge(merged_data, data_inflation, on='Year')
merged_data = pd.merge(merged_data, data_employment, on='Year')
merged_data = pd.merge(merged_data, data_trade, on='Year')

merged_data = pd.merge(merged_data, data_firstMarriageRatio, on='Year')
# merged_data = pd.merge(merged_data, data_HICP, on='Year')
# merged_data = pd.merge(merged_data, data_consumerIndexes, on='Year')
merged_data = pd.merge(merged_data, data_salaryIndexes, on='Year')

merged_data = merged_data[merged_data['Year'].astype(int) > 1995]


print(merged_data.columns)
# provjeri sa i bez Year, provjerit sve kombinacije moguce ale ale ala, kaj je najmanje moguce a da dela dosta dobro?
# jos neke moguce opcije(nisu pomogle sa svim ovim ali sa manjim skupom mozda bi): Marriage_to_Population_Ratio mrvicu odmoglo but idk, od indeksa placa nic ne pomaze :(
#  % of GDP-Imports | Exports-Billions of US $ | % of GDP-Exports || trade_balance || trade_ratio | Unemployment Rate (%) |  Unemployment_Annual Change | 
x = merged_data[['Net migration',  
                'Rural population growth (annual %)', 
                'Population in the largest city (% of urban population)',
                'Exchange rate, new LCU per USD extended backward, period average,,',
                'CPI Price, seas. adj.,,,',
                'CPI Price,not seas.adj,,,',
                'Age dependency ratio, young',  
                'Population ages 15-64 (% of total population)',
                'Population ages 20-24, female (% of female population)',
                '% of GDP', 'Inflation Rate (%)', '% of Total Labor Force Ages 15-24', 
                'Imports-Billions of US $']]
y = merged_data['Births']

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



