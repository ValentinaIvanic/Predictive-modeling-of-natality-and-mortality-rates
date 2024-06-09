from model_imports import *
from sklearn.linear_model import LinearRegression

#--------------------------------------------- << Data + Model >> ------------------------------------------------

data = Data_extracting.get_worldbankForBirths()
data_births = Data_extracting.get_births()
data_gdp = Data_extracting.get_maddisonProjectData()

merged_data = pd.merge(data, data_births, on='Year')
merged_data = pd.merge(merged_data, data_gdp, on='Year')
merged_data = merged_data.dropna(axis=0, how='any')
x = merged_data[['Year', 'Net migration',  
                'Population in the largest city (% of urban population)', 'Rural population growth (annual %)',  
                'Population ages 15-64 (% of total population)',
                'Population ages 20-24, female (% of female population)', 'GDP_per_capita_2011_prices']].values  


# x = merged_data[['Year', 
#                 'GDP_per_capita_2011_prices']].values  
y = merged_data['Births'].values.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=100)

model = LinearRegression()
model.fit(x_train, y_train)

#--------------------------------------------- << Results >> ------------------------------------------------

from check_score_metrics import *
import  matplotlib.pyplot as plt
import seaborn as sns

y_pred = model.predict(x_test)
print_scores(y_test, y_pred)

coefficients = pd.DataFrame(model.coef_[0], index=['Net migration',  
                'Population in the largest city (% of urban population)', 'Rural population growth (annual %)',  
                'Population ages 15-64 (% of total population)',
                'Population ages 20-24, female (% of female population)', 'GDP_per_capita_2011_prices'], columns=['Coefficient'])

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y=coefficients.index, data=coefficients, palette='viridis')
plt.title('Coefficients of Features')
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.show()


#--------------------------------------------- << Predict new data >> ------------------------------------------------

# current_year = merged_data['Year'].max()
# years_to_predict = np.arange(current_year + 1, current_year + 51).reshape(-1, 1)

# births_predicted = model.predict(years_to_predict)

# #save predicted data

# data_predicted = pd.DataFrame({
#     'Year': years_to_predict.flatten(),
#     'Births': births_predicted.flatten()
# })

# data_predicted['Births'] = data_predicted['Births'].astype(int)


# data_predicted.to_csv('Data/Predicted/births_predicted.csv', index = False)



# #--------------------------------------------- << TimeSeriesSplit >> ------------------------------------------------
# print("Time series split")
# from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# model_TSS = LinearRegression()

# TSS = TimeSeriesSplit(n_splits=3)

# scores = cross_val_score(model_TSS, x, y, cv=TSS, scoring='r2')

# print(scores)

