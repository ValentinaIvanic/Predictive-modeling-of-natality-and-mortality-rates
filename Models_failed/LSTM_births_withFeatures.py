from model_imports import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt


data_births = Data_extracting.get_births()
data_features = Data_extracting.get_Births_for_LSTM()


data_bestFeatures = data_features[['Year', 'Net migration',  
                                    'Population in the largest city (% of urban population)', 
                                    'Rural population growth (annual %)']]

# print(data.info())

data_births.set_index('Year', inplace=True)
data_bestFeatures.set_index('Year', inplace=True)
# print(data.info())
print(data_bestFeatures)
print(data_births)
# print(data)

births = data_births['Births']


#izvor orginalne funkcije (df_to_X_y()): https://www.youtube.com/watch?v=c0k-YLQGKjY&ab_channel=GregHogg , https://colab.research.google.com/drive/1HxPsJvEAH8L7XTmLnfdJ3UQx7j0o1yX5?usp=sharing
def df_to_X_y_with_features(df_births, df_features, window_size):
    df_as_np_births = df_births.to_numpy()
    df_as_np_features = df_features.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np_births) - window_size):
        rows = [np.hstack((df_as_np_births[i: i + window_size], df_as_np_features[i + window_size]))]
        X.append(rows)
        label = df_as_np_births[i + window_size]
        y.append(label)
    return np.array(X), np.array(y)

scaler = MinMaxScaler()
scaled_births = scaler.fit_transform(births.values.reshape(-1, 1))

scaler_economic = StandardScaler()
scaled_features = scaler_economic.fit_transform(data_bestFeatures.values)

X, y = df_to_X_y_with_features(pd.Series(scaled_births.flatten(), index=births.index), pd.DataFrame(scaled_features, index=data_bestFeatures.index),  10)

print(X.shape)
print(y.shape)

x_train, y_train = X[:45], y[:45]
x_val, y_val = X[45:50], y[45:50]
x_test, y_test = X[50:], y[50:]



print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
print(y_test.shape)

model1 = Sequential()
model1.add(InputLayer((10, scaled_features.shape[1] + 1)))
model1.add(LSTM(256))
model1.add(Dense(16, 'relu'))
model1.add(Dense(1, 'linear'))

model1.summary()

cp = ModelCheckpoint('model_new.keras', save_best_only=True)
model1.compile(loss = MeanSquaredError(), optimizer = Adam(learning_rate=0.01), metrics=[RootMeanSquaredError()])

model1.fit(x_train, y_train, validation_data = (x_val, y_val), epochs = 500, callbacks = [cp])

model1 = load_model('model_new.keras')

train_predict = model1.predict(x_train).flatten()
val_predict = model1.predict(x_val).flatten()
test_predict = model1.predict(x_test).flatten()


train_results = pd.DataFrame(data={'Train Predictions':train_predict, 'Actuals':y_train})
print(train_results)

train_predict = model1.predict(x_train)
train_predict = scaler.inverse_transform(train_predict)
train_predict = train_predict.flatten()

val_predict = model1.predict(x_val)
val_predict = scaler.inverse_transform(val_predict)
val_predict = val_predict.flatten()

test_predict = model1.predict(x_test)
test_predict = scaler.inverse_transform(test_predict)
test_predict = test_predict.flatten()


y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
y_train = y_train.flatten()

y_val = scaler.inverse_transform(y_val.reshape(-1, 1))
y_val = y_val.flatten()

y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
y_test = y_test.flatten()



train_results = pd.DataFrame(data={'Train Predictions':train_predict, 'Actuals':y_train})
print(train_results)

val_results = pd.DataFrame(data={'Val Predictions':val_predict, 'Actuals':y_val})
print(val_results)

test_results = pd.DataFrame(data={'Test Predictions':test_predict, 'Actuals':y_test})
print(test_results)

print(x_test)


from check_score_metrics import *

print_scores(y_train, train_predict)
print_scores(y_val, val_predict)
print_scores(y_test, test_predict)


plt.figure(figsize=(10,6))
plt.plot(train_results['Train Predictions'])
plt.plot(train_results['Actuals'])
plt.xlabel('Godina')
plt.ylabel('Broj rođenih')
plt.title('Grafikon broja rođenih po godini')
plt.show()


last_window = scaled_births[-10:]
future_predictions = []

for _ in range(2023, 2051):
    prediction = model1.predict(last_window.reshape(1, 10, 1))
    future_predictions.append(prediction[0][0])
    last_window = np.append(last_window[1:], prediction)[np.newaxis].T

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

years_future = np.arange(2023, 2051)
plt.figure(figsize=(10,6))
plt.plot(data.index, births, label='Historical Data')
plt.plot(years_future, future_predictions, label='Future Predictions', linestyle='dashed')
plt.xlabel('Year')
plt.ylabel('Births')
plt.title('Birth Predictions from 2023 to 2050')
plt.legend()
plt.show()