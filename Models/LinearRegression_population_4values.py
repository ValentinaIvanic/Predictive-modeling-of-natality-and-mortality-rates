from model_imports import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

#--------------------------------------------- << Data + Model >> ------------------------------------------------

data_population = Data_extracting.get_population()
data_births = Data_extracting.get_births()
data_deaths = Data_extracting.get_deaths()

merged_data = pd.merge(data_population, data_births, on='Year')
merged_data = pd.merge(merged_data, data_deaths, on='Year')

# print(merged_data.head())

# x = data_population['Year'].values.reshape(-1, 1) #pretvorba iz DataFrame-a u 2d nizove
# y = data_population['Population'].values.reshape(-1, 1)

x = merged_data[['Year', 'Births', 'Deaths']].values  
y = merged_data['Population'].values.reshape(-1, 1)  

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.63, random_state=100)

model = LinearRegression()
model.fit(x_train, y_train)


#--------------------------------------------- << Results >> ------------------------------------------------

from check_score_metrics import *

y_pred = model.predict(x_test)
print_scores(y_test, y_pred)

print(y_pred)
print("--------------------------------")


#--------------------------------------------- << Predict new data >> ------------------------------------------------

data_predictedBirths = Data_extracting.get_predictedBirths()
data_predictedDeaths = Data_extracting.get_predictedDeaths()

year_from = merged_data['Year'].max()
years_to_predict = np.arange(year_from + 1, year_from + 51).reshape(-1, 1)

merged_predictedData = pd.merge(data_predictedBirths, data_predictedDeaths, on = 'Year')

data_predictedPopulation = model.predict(merged_predictedData)

print(merged_data.tail())
print("--------------------------------")
print(data_predictedBirths.head())
print("--------------------------------")
print(data_predictedDeaths.head())
print("--------------------------------")
print(data_predictedPopulation)


#---------------------------------------------------------<< Ridge >>-------------------------------------------------------------


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.63, random_state=100)


ridge_model = Ridge(alpha=1.0)
ridge_model.fit(x_train, y_train)

# Predict and evaluate the Ridge model
y_pred_ridge = ridge_model.predict(x_test)
print_scores(y_test, y_pred_ridge)


#--------------------------------------------------------<< PolynominalFeatures >>-------------------------------------------------

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(x)

x_train_poly, x_test_poly, y_train, y_test = train_test_split(x_poly, y, train_size=0.6, random_state=100)

model_poly = LinearRegression()
model_poly.fit(x_train_poly, y_train)

y_pred_poly = model_poly.predict(x_test_poly)
print_scores(y_test, y_pred_poly)

print("-------------------------------------")
print(y_pred_poly)

print("-------------------------------------")
print(y_test)


merged_predictedData_POLY = poly.transform(x)
data_predictedPopulation_POLY = model_poly.predict(merged_predictedData_POLY)

print(merged_data.tail())
print("--------------------------------")
print(data_predictedBirths.head())
print("--------------------------------")
print(data_predictedDeaths.head())
print("--------------------------------")
print(data_predictedPopulation_POLY)
