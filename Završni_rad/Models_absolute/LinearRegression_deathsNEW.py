from model_imports import *
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
#--------------------------------------------- << Data + Model >> ------------------------------------------------

data = Data_extracting.get_worldbankForDeaths()
data_deaths = Data_extracting.get_deaths()
data_gdp = Data_extracting.get_maddisonProjectData()

merged_data = pd.merge(data, data_deaths, on='Year')
merged_data = pd.merge(merged_data, data_gdp, on='Year')
merged_data = merged_data.dropna(axis=0, how='any')
 

scaler = StandardScaler()
scaled_features = scaler.fit_transform(merged_data[['Year', 
                'Life expectancy at birth, total (years)', 
                'Age dependency ratio, old', 
                'Survival to age 65, male (% of cohort)']])

scaled_df = pd.DataFrame(scaled_features, columns=['Year', 
                'Life expectancy at birth, total (years)', 
                'Age dependency ratio, old', 
                'Survival to age 65, male (% of cohort)'])

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
                'Age dependency ratio, old', 
                'Survival to age 65, male (% of cohort)'], columns=['Coefficient'])

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y=coefficients.index, data=coefficients, palette='viridis')
plt.title('Koeficijenti varijabli za model', fontsize = 17)
plt.xlabel('Vrijednost koeficijenta',  fontsize = 15)
plt.ylabel('Varijable',  fontsize = 15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.show()

