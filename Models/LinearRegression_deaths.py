from model_imports import *
from sklearn.linear_model import LinearRegression

#--------------------------------------------- << Data + Model >> ------------------------------------------------

data_deaths = Data_extracting.get_deaths()

x = data_deaths['Year'].values.reshape(-1, 1) #pretvorba iz DataFrame-a u 2d nizove
y = data_deaths['Deaths'].values.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=100)

model = LinearRegression()
model.fit(x_train, y_train)


 
#--------------------------------------------- << Results >> ------------------------------------------------

from check_score_metrics import *

y_pred = model.predict(x_test)
print_scores(y_test, y_pred)



#--------------------------------------------- << Predict new data >> ------------------------------------------------

current_year = data_deaths['Year'].max()
years_to_predict = np.arange(current_year + 1, current_year + 51).reshape(-1, 1)

deaths_predicted = model.predict(years_to_predict)

#save predicted data

data_predicted = pd.DataFrame({
    'Year': years_to_predict.flatten(),
    'Deaths': deaths_predicted.flatten()
})

data_predicted['Deaths'] = data_predicted['Deaths'].astype(int)

data_predicted.to_csv('Data/Predicted/deaths_predicted.csv', index = False)

