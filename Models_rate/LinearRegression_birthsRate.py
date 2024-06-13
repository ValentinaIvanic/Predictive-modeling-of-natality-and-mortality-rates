from model_imports import *
from sklearn.linear_model import LinearRegression

#--------------------------------------------- << Data + Model >> ------------------------------------------------

data = Data_extracting.get_worldbankForBirths()
data_births = Data_extracting.get_births()
data_gdp = Data_extracting.get_maddisonProjectData()

merged_data = pd.merge(data, data_births, on='Year')
merged_data = pd.merge(merged_data, data_gdp, on='Year')
merged_data = merged_data.dropna(axis=0, how='any')

#---------------------------------------------------------------------------> BEZ BDP-a
x = merged_data[['Year', 
                'Population in the largest city (% of urban population)', 'Rural population growth (annual %)',  
                'Population ages 15-64 (% of total population)','GDP_per_capita_2011_prices']].values  

y = merged_data['Birth rate, crude (per 1,000 people)'].values.reshape(-1, 1)

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
                'Population in the largest city (% of urban population)', 'Rural population growth (annual %)',  
                'Population ages 15-64 (% of total population)','GDP_per_capita_2011_prices'], columns=['Coefficient'])

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y=coefficients.index, data=coefficients, palette='viridis')
plt.title('Coefficients of Features')
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.show()
