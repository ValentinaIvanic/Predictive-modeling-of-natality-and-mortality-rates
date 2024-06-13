from model_imports import *
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt
import plotly.graph_objects as go

#----------------------------------------------------<< Funkcije za grafove >>-----------------------------------------------

# Izvor funkcije plot_func: https://github.com/egorhowell/Youtube/blob/main/Time-Series-Crash-Course/11.%20Holt%27s%20Linear%20Trend%20Model.ipynb
def plot_func(forecast: list[float], title: str) -> None:
    """Function to plot the forecasts."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train['Year'], y=train['Death rate, crude (per 1,000 people)'], name='Train'))
    fig.add_trace(go.Scatter(x=test['Year'], y=test['Death rate, crude (per 1,000 people)'], name='Test'))
    fig.add_trace(go.Scatter(x=test['Year'], y=forecast, name='Forecast'))
    fig.update_layout(template="simple_white", font=dict(size=18), title_text=title,
                      width=650, title_x=0.5, height=400, xaxis_title='Year',
                      yaxis_title='Deaths')
    
    return fig.show()


# Izvor plot_func_HOLT: https://github.com/egorhowell/Youtube/blob/main/Time-Series-Crash-Course/11.%20Holt's%20Linear%20Trend%20Model.ipynb

def plot_func_HOLT(train, test, forecast1: list[float],
              forecast2: list[float],
              title: str, stupac, naslov) -> None:
    """Function to plot the forecasts."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train['Year'], y=train[stupac], name='Train'))
    fig.add_trace(go.Scatter(x=test['Year'], y=test[stupac], name='Test'))
    fig.add_trace(go.Scatter(x=test['Year'], y=forecast1, name='Exp. Smoothing'))
    fig.add_trace(go.Scatter(x=test['Year'], y=forecast2, name='Holt'))
    fig.update_layout(template="simple_white", font=dict(size=15), title_text=title,
                      width=650, title_x=0.5, height=400, xaxis_title='Godina',
                      yaxis_title= naslov)

    return fig.show()

#-------------------------------------------------------<< Exponential Smoothing Model >>------------------------------------------------------


def exp_Smoothing(train, test, stupac, trend_mod, damped_mod):
    model_ExpSmoothing = ExponentialSmoothing(train[stupac], trend=trend_mod, damped_trend = damped_mod).fit()
    forecast = model_ExpSmoothing.forecast(len(test))
    print("####################################################################")
    print(model_ExpSmoothing.summary())
    print("####################################################################")
    return forecast


#-------------------------------------------------------<< Holt's Linear Trend Model >>------------------------------------------------------


def model_Holt(train, test, stupac, damped_mod):
    model= Holt(train[stupac], damped_trend=damped_mod).fit()
    forecast_Holt = model.forecast(len(test))
    print("####################################################################")
    print(model.summary())
    print("####################################################################")
    return forecast_Holt

#--------------------------------------------------------<< Deaths >>----------------------------------------------------------------------


data = Data_extracting.smoothingHolt_deaths()
# ['Year', 'Life expectancy at birth, total (years)', 'Urban population (% of total population)', 
#                       'Urban population', 'Survival to age 65, female (% of cohort)', 'Survival to age 65, male (% of cohort)', 
#                       'Rural population', 'Rural population (% of total population)', 'Rural population growth (annual %)', 
#                       'Population, total', 'Net migration', 'Population in largest city', 'Population growth (annual %)', 
#                       'Population ages 80 and above, male (% of male population)', 'Population in the largest city (% of urban population)',
#                       'Death rate, crude (per 1,000 people)']


train_deaths = data.iloc[:int(0.6 * len(data))]
test_deaths = data.iloc[int(0.6 * len(data)):]


forecast_expSmoothing_Deaths = exp_Smoothing(train_deaths, test_deaths, 'Death rate, crude (per 1,000 people)', 'add', False)
forecast_Holt_deaths = model_Holt(train_deaths, test_deaths, 'Death rate, crude (per 1,000 people)', False)

plot_func_HOLT(train_deaths, test_deaths, forecast_expSmoothing_Deaths, forecast_Holt_deaths, "Modeli s izglađivanjem", "Death rate, crude (per 1,000 people)", "Broj umrlih na 1000 ljudi")

#--------------------------------------------------------<< Births >>----------------------------------------------------------------------

data = Data_extracting.smoothingHolt_births()
# ['Year', 'Net migration', 'Population in largest city', 'Population growth (annual %)', 'Population, total', 
#                       'Rural population', 'Rural population (% of total population)', 'Rural population growth (annual %)', 
#                       'Urban population (% of total population)', 'Urban population', 'Population in the largest city (% of urban population)',
#                       'Birth rate, crude (per 1,000 people)']


data_births = Data_extracting.get_births()


train = data.iloc[:int(0.5 * len(data))]
test = data.iloc[int(0.5 * len(data)):]


forecast_expSmoothing = exp_Smoothing(train, test, 'Birth rate, crude (per 1,000 people)', 'add', True)
forecast_Holt = model_Holt(train, test, 'Birth rate, crude (per 1,000 people)', True)

plot_func_HOLT(train, test, forecast_expSmoothing, forecast_Holt, "Modeli s izglađivanjem", "Birth rate, crude (per 1,000 people)", "Broj rođenih na 1000 ljudi")


from check_score_metrics import *



print("Births expSmoothing")
print_scores(test['Birth rate, crude (per 1,000 people)'].iloc[:-2], forecast_expSmoothing.iloc[:-2])
print("-------------------------------------------")

print("Births Holt")
print_scores(test['Birth rate, crude (per 1,000 people)'].iloc[:-2], forecast_Holt.iloc[:-2])
print("-------------------------------------------")

print("Deaths expSmoothing")
print_scores(test_deaths['Death rate, crude (per 1,000 people)'].iloc[:-2], forecast_expSmoothing_Deaths.iloc[:-2])
print("-------------------------------------------")

print("Deaths Holt")
print_scores(test_deaths['Death rate, crude (per 1,000 people)'].iloc[:-2], forecast_Holt_deaths.iloc[:-2])
print("-------------------------------------------")