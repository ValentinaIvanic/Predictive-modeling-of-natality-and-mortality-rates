from model_imports import *
from sklearn.linear_model import LinearRegression

#--------------------------------------------- << Data + Model >> ------------------------------------------------

data = Data_extracting.get_worldbankForDeaths()
data_deaths = Data_extracting.get_deaths()
data_gdp = Data_extracting.get_maddisonProjectData()

merged_data = pd.merge(data, data_deaths, on='Year')
merged_data = pd.merge(merged_data, data_gdp, on='Year')
merged_data = merged_data.dropna(axis=0, how='any')



x = merged_data[['Year', 
                'GDP_per_capita_2011_prices' ,'Life expectancy at birth, total (years)', 
                'Urban population (% of total population)', 
                'Survival to age 65, female (% of cohort)', 'Survival to age 65, male (% of cohort)', 
                'Rural population (% of total population)', 'Rural population growth (annual %)', 
                'Net migration', 'Population growth (annual %)', 
                'Population ages 80 and above, male (% of male population)', 
                'Population in the largest city (% of urban population)',
                'Population ages 65 and above (% of total population)',
                'Population ages 15-64 (% of total population)',
                'Age dependency ratio, old']].values  

y = merged_data['Deaths'].values.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.78, random_state=100)

model = LinearRegression()
model.fit(x_train, y_train)


 
#--------------------------------------------- << Results >> ------------------------------------------------
from check_score_metrics import *
import  matplotlib.pyplot as plt
import seaborn as sns

y_pred = model.predict(x_test)
print_scores(y_test, y_pred)

coefficients = pd.DataFrame(model.coef_[0], index=['Year', 
                'GDP_per_capita_2011_prices' ,'Life expectancy at birth, total (years)', 
                'Urban population (% of total population)', 
                'Survival to age 65, female (% of cohort)', 'Survival to age 65, male (% of cohort)', 
                'Rural population (% of total population)', 'Rural population growth (annual %)', 
                'Net migration', 'Population growth (annual %)', 
                'Population ages 80 and above, male (% of male population)', 
                'Population in the largest city (% of urban population)',
                'Population ages 65 and above (% of total population)',
                'Population ages 15-64 (% of total population)',
                'Age dependency ratio, old'], columns=['Coefficient'])

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y=coefficients.index, data=coefficients, palette='viridis')
plt.title('Coefficients of Features')
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.show()


#--------------------------------------------- << Predict new data >> ------------------------------------------------

# current_year = data_deaths['Year'].max()
# years_to_predict = np.arange(current_year + 1, current_year + 51).reshape(-1, 1)

# deaths_predicted = model.predict(years_to_predict)

# #save predicted data

# data_predicted = pd.DataFrame({
#     'Year': years_to_predict.flatten(),
#     'Deaths': deaths_predicted.flatten()
# })

# data_predicted['Deaths'] = data_predicted['Deaths'].astype(int)

# data_predicted.to_csv('Data/Predicted/deaths_predicted.csv', index = False)

