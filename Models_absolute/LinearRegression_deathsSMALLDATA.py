from model_imports import *
from sklearn.linear_model import LinearRegression

#--------------------------------------------- << Data + Model >> ------------------------------------------------

data = Data_extracting.get_worldbankForDeaths()
data_deaths = Data_extracting.get_deaths()
data_gdp = Data_extracting.get_maddisonProjectData()
data_economics = Data_extracting.get_worldbankEconomics()
data_inflation = Data_extracting.get_inflation() #kvari

merged_data = pd.merge(data, data_deaths, on='Year')
merged_data = pd.merge(merged_data, data_gdp, on='Year')
merged_data = pd.merge(merged_data, data_economics, on='Year')
merged_data = pd.merge(merged_data, data_inflation, on='Year')

merged_data = merged_data[merged_data['Year'].astype(int) > 1986]
merged_data = merged_data.dropna(axis=0, how='any')

# draw_Graphs.variable_by_years(merged_data['Year'], merged_data['Deaths'], 'Deaths')

# x = merged_data[['Year', 
#                 'GDP_per_capita_2011_prices' ,'Life expectancy at birth, total (years)', 
#                 'Urban population (% of total population)', 
#                 'Survival to age 65, female (% of cohort)', 'Survival to age 65, male (% of cohort)', 
#                 'Rural population (% of total population)', 'Rural population growth (annual %)', 
#                 'Net migration', 'Population growth (annual %)', 
#                 'Population ages 80 and above, male (% of male population)', 
#                 'Population in the largest city (% of urban population)',
#                 'Population ages 65 and above (% of total population)',
#                 'Population ages 15-64 (% of total population)',
#                 'Age dependency ratio, old']].values  

# ------------------- Probaj bez survival to age 65?, sa 'Survival to age 65, female (% of cohort)', 'Survival to age 65, male (% of cohort)',  je poprilican cheating
# x = merged_data[['Year', 
#                 'Life expectancy at birth, total (years)', 
#                 'Age dependency ratio, old', 
#                 'Survival to age 65, male (% of cohort)',

#                 'Exchange rate',
#                 'CPI Price, seasonal',
#                 'CPI Price']].values  

scaler = StandardScaler()
scaled_features = scaler.fit_transform(merged_data[['Year', 
                'Life expectancy at birth, total (years)', 
                'Survival to age 65, male (% of cohort)',

                'Exchange rate',
                'CPI Price, seasonal',
                'CPI Price']])

scaled_df = pd.DataFrame(scaled_features, columns=['Year', 
                'Life expectancy at birth, total (years)', 
                'Survival to age 65, male (% of cohort)',

                'Exchange rate',
                'CPI Price, seasonal',
                'CPI Price'])

x = scaled_df.values


y = merged_data['Deaths'].values.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=100, shuffle=False)

model = LinearRegression()
model.fit(x_train, y_train)

 
#--------------------------------------------- << Results >> ------------------------------------------------
from check_score_metrics import *
import  matplotlib.pyplot as plt
import seaborn as sns

test_pred = model.predict(x_test)
train_pred = model.predict(x_train)
print_scores(y_test, test_pred)
print_scores(y_train, train_pred)

draw_Graphs.train_test_testPred(merged_data['Year'], y_train, y_test, test_pred, len(y_train))
draw_Graphs.train_test_trainpred_testPred(merged_data['Year'], y_train, y_test, test_pred, train_pred, len(y_train))

coefficients = pd.DataFrame(model.coef_[0], index=['Year', 
                'Life expectancy at birth, total (years)', 
                'Survival to age 65, male (% of cohort)',
                'Exchange rate',
                'CPI Price, seasonal',
                'CPI Price'], columns=['Coefficient'])

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y=coefficients.index, data=coefficients, palette='viridis')
plt.title('Koeficijenti varijabli za model', fontsize = 17)
plt.xlabel('Vrijednost koeficijenta',  fontsize = 15)
plt.ylabel('Varijable',  fontsize = 15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.show()