from model_imports import *
from sklearn.linear_model import LinearRegression

#--------------------------------------------- << Data + Model >> ------------------------------------------------

data = Data_extracting.get_worldbankForBirths()
data_births = Data_extracting.get_births()
data_gdp = Data_extracting.get_maddisonProjectData()
data_economics = Data_extracting.get_worldbankEconomics()
data_inflation = Data_extracting.get_inflation() #kvari

merged_data = pd.merge(data, data_births, on='Year')
merged_data = pd.merge(merged_data, data_gdp, on='Year')
merged_data = pd.merge(merged_data, data_economics, on='Year')
merged_data = pd.merge(merged_data, data_inflation, on='Year')

merged_data = merged_data.dropna(axis=0, how='any')
merged_data = merged_data[merged_data['Year'].astype(int) > 1986]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(merged_data[['Year', 'Exchange rate',
                'CPI Price, seasonal',
                'Age dependency ratio, young', 
                'Population in the largest city (% of urban population)',   
                'Population ages 15-64 (% of total population)','GDP_per_capita_2011_prices']])

scaled_df = pd.DataFrame(scaled_features, columns=['Year', 'Exchange rate',
                'CPI Price, seasonal',
                'Age dependency ratio, young', 
                'Population in the largest city (% of urban population)',   
                'Population ages 15-64 (% of total population)','GDP_per_capita_2011_prices'])

x = scaled_df.values


y = merged_data['Births'].values.reshape(-1, 1)

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


coefficients = pd.DataFrame(model.coef_[0], index=['Year', 'Exchange rate',
                'CPI Price, seasonal',
                'Age dependency ratio, young', 
                'Population in the largest city (% of urban population)',   
                'Population ages 15-64 (% of total population)','GDP_per_capita_2011_prices'], columns=['Coefficient'])

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y=coefficients.index, data=coefficients, palette='viridis')
plt.title('Koeficijenti varijabli za model', fontsize = 17)
plt.xlabel('Vrijednost koeficijenta',  fontsize = 15)
plt.ylabel('Varijable',  fontsize = 15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.show()
