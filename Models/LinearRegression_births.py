from model_imports import *
from sklearn.linear_model import LinearRegression

#--------------------------------------------- << Data + Model >> ------------------------------------------------

data_births = Data_extracting.get_births()

x = data_births['Year'].values.reshape(-1, 1) #pretvorba iz DataFrame-a u 2d nizove
y = data_births['Births'].values.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=100)

model = LinearRegression()
model.fit(x_train, y_train)

#--------------------------------------------- << Results >> ------------------------------------------------

from check_score_metrics import *

y_pred = model.predict(x_test)
print_scores(y_test, y_pred)


#--------------------------------------------- << Predict new data >> ------------------------------------------------

current_year = data_births['Year'].max()
years_to_predict = np.arange(current_year + 1, current_year + 51).reshape(-1, 1)

births_predicted = model.predict(years_to_predict)

#save predicted data

data_predicted = pd.DataFrame({
    'Year': years_to_predict.flatten(),
    'Births': births_predicted.flatten()
})

data_predicted['Births'] = data_predicted['Births'].astype(int)


data_predicted.to_csv('Data/Predicted/births_predicted.csv', index = False)



#--------------------------------------------- << TimeSeriesSplit >> ------------------------------------------------
print("Time series split")
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

model_TSS = LinearRegression()

TSS = TimeSeriesSplit(n_splits=3)

scores = cross_val_score(model_TSS, x, y, cv=TSS, scoring='r2')

print(scores)

