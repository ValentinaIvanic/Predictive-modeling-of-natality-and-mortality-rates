import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from Data_preprocessing import Data_extracting

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#--------------------------------------------- << Data + Model >> ------------------------------------------------

data_births = Data_extracting.get_deaths()

x = data_births['Year'].values.reshape(-1, 1) #pretvorba iz DataFrame-a u 2d nizove
y = data_births['Deaths'].values.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=100)

model = LinearRegression()
model.fit(x_train, y_train)

#--------------------------------------------- <</ Data + Model >> ------------------------------------------------


#--------------------------------------------- << Results >> ------------------------------------------------

from check_score_metrics import *

y_pred = model.predict(x_test)
print_scores(y_test, y_pred)

#--------------------------------------------- <</ Results >> ------------------------------------------------


