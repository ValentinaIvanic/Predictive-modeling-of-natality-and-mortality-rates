import pandas as pd

pd.set_option('display.max_rows', None)  # Prikaži sve retke
pd.set_option('display.max_columns', None)  # Prikaži sve stupce


def get_dataFromEurostat(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, skiprows=7)

    filtered_df = df[df[0].isin(['TIME', 'Croatia'])]

    filtered_df = filtered_df[~filtered_df[0].isna()]
    filtered_df = filtered_df.dropna(axis=1, how='all')

    data_transposed = filtered_df.T
    data_transposed.columns = data_transposed.iloc[0]
    data_transposed = data_transposed[1:].reset_index(drop=True)
    data_transposed = data_transposed.dropna(subset=['TIME'], axis=0)

    # print(data_transposed)
    data_transposed.rename(columns={'TIME': 'Year', 'Croatia': 'Population'}, inplace=True)


    return data_transposed

def get_dataFromHNB(file_path, sheet_name, header):
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=header, skiprows=1)

    df.columns = df.iloc[0]
    df = df.drop(0)

    df.reset_index(drop=True, inplace=True)
    filtered_df = df.dropna(how='all')

    
    # print(filtered_df)

    return filtered_df


def data_to_int(data):
    data['Year'] = data['Year'].astype(int)
    data['Population'] = data['Population'].astype(int)
    return data

# --------------------------------------------<< Data from Eurostat >> ----------------------------------------------------


def get_deaths():
    data = get_dataFromEurostat("Data\Original\Deaths (total) by month.xlsx", "Sheet 1")
    data = data.drop(data.index[-1])
    data = data_to_int(data)
    data.rename(columns={'Population': 'Deaths'}, inplace=True)
    return data

def get_births():
    data = get_dataFromEurostat("Data\Original\Live births (total) by month.xlsx", "Sheet 1")
    data = data.drop(data.index[-1])
    data = data_to_int(data)
    data.rename(columns={'Population': 'Births'}, inplace=True)
    return data

def get_population():
    data = get_dataFromEurostat("Data\Original\Population on 1 January by age and sex.xlsx", "Sheet 1")
    data = data_to_int(data)
    return data

#-------

def get_womenAgeFirstBirth():
    return get_dataFromEurostat("Data\Original\Fertility indicators - mean women age at first child birth.xlsx", "Sheet 1")

def get_womenCompletedAgeFirstMarriage():
    return get_dataFromEurostat("Data\Original\First marriage rates by age and sex - females.xlsx", "Sheet 1")

def get_manCompletedAgeFirstMarriage():
    return get_dataFromEurostat("Data\Original\First marriage rates by age and sex - males.xlsx", "Sheet 1")

def get_ratioFirstTimeMarriedToPopulation():
    df = get_dataFromEurostat("Data\Original\First-time marrying persons by age and sex.xlsx", "Sheet 1")
    df.rename(columns={'Population': 'Number of first-time married in a year'}, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df['Year'] = df['Year'].astype(int)
    data_population = get_population()
    merged_df = pd.merge(df, data_population, on='Year', how='inner')
    merged_df['Marriage_to_Population_Ratio'] = (merged_df['Number of first-time married in a year'] / merged_df['Population']) * 100

    selected_columns = merged_df.loc[:, ['Year', 'Marriage_to_Population_Ratio']]
    return selected_columns

def get_HICP_Eurostat():
    df = pd.read_excel("Data\Original\Harmonised index of consumer prices mjesečno.xlsx", sheet_name="Sheet 1", header=None, skiprows=7)

    filtered_df = df[df[0].isin(['TIME', 'Croatia'])]

    filtered_df = filtered_df[~filtered_df[0].isna()]
    filtered_df = filtered_df.dropna(axis=1, how='all')

    data_transposed = filtered_df.T
    data_transposed.columns = data_transposed.iloc[0]
    data_transposed = data_transposed[1:].reset_index(drop=True)
    data_transposed = data_transposed.dropna(subset=['TIME'], axis=0)

    data_transposed.rename(columns={'TIME': 'Year', 'Croatia': 'HICP'}, inplace=True)
    data_transposed = data_transposed.iloc[24:]


    data_transposed['Year'] = pd.to_datetime(data_transposed['Year'])
    data_transposed['Year'] = data_transposed['Year'].dt.year

    df = df.apply(pd.to_numeric, errors='coerce')
    average_by_year = data_transposed.groupby('Year').mean()

    return average_by_year

def get_HICPbyMonth():
    return get_dataFromEurostat("Data\Original\Harmonised index of consumer prices mjesečno.xlsx", "Sheet 1")

# --------------------------------------------<</ Data from Eurostat >> ----------------------------------------------------

# --------------------------------------------<< Data from HNB >> ----------------------------------------------------

def get_PriceIndexResidentalBuilding():
    df = get_dataFromHNB("Data\Original\Indeksi cijena stambenih objekata.xlsx", "HRV", 2)
    df = df.iloc[:-4]
    print(df)
    return df
def get_ConsumerProducerPricesIndex():
    df = get_dataFromHNB("Data\Original\Indeksi potrošačkih cijena i prozivođačkih cijena industrije.xlsx", "HRV", 2)
    df = df.iloc[:-3]
    print(df)
    return df

def get_FundamentalConsumerPriceIndexes():
    df = get_dataFromHNB("Data\Original\Temeljni indeksi potrošačkih cijena.xlsx", "HRV", 4)
    df = df.iloc[:-2]
    print(df)
    return df

def get_HICP():
    df = get_dataFromHNB("Data\Original\Harmonizirani indeksi potrošačkih cijena.xlsx", "HRV", 2)
    df = df.iloc[:-1]
    print(df)
    return df

def get_AverageSalaryByMonth():
    df = pd.read_excel("Data\Original\Indeksi_placa.xlsx", usecols=['Year', 'Godišnji lančani indeks', 'Godišnji mjesešni indeksi average', 'Godišnji kumulativni indeksi average'], nrows = 31)
    df = df.apply(pd.to_numeric, errors='coerce')
    df['Year'] = df['Year'].astype(int)

    return df


def get_ConsumerIndexes():
    df = pd.read_excel("Data\Original\Indeks_potrosaca.xlsx", usecols=['Year', 'Indeks_pouzdanja_potrosaca', 'Indeks_ocekivanja_potrosaca', 'Indeks_raspolozenja_potrosaca'], nrows = 26)
    df = df.apply(pd.to_numeric, errors='coerce')
    df['Year'] = df['Year'].astype(int)

    return df


# --------------------------------------------<</ Data from HNB >> ----------------------------------------------------
#------------------------------------------------- << New predicted data >> -------------------------------------------

def get_predictedBirths():
    data = pd.read_csv("Data/Predicted/births_predicted.csv")
    return data

def get_predictedDeaths():
    data = pd.read_csv("Data/Predicted/deaths_predicted.csv")
    return data

#------------------------------------------------- << Data from WorldBank >> -------------------------------------------

def worldbankData_transform():

    data = pd.read_excel("Data/Original/P_Data_Extract_From_World_Development_Indicators.xlsx", sheet_name="Data")

    data = data.drop(data.columns[[1, 2, 3]], axis=1)
    data = data.T

    data.rename(columns={'Series Name': 'Year'}, inplace=True)
    print("------------------------")

    data.reset_index(inplace=True)
    data.insert(0, 'Series Name', data.pop('index'))
    data.reset_index(drop=True, inplace=True)


    new_column_names = data.iloc[0]

    data.columns = new_column_names
    data = data.drop(0)
    data.reset_index(drop=True, inplace=True)
    data.rename(columns={'Series Name': 'Year'}, inplace=True)
    data['Year'] = data['Year'].str.extract('(\d{4})')

    data.to_excel('Data/Original/WorldBank_transformed.xlsx', index = False)

def worldBank_populationsEstimates():
    data2 = pd.read_excel("Data/Original/P_Data_Extract_From_Population_estimates_and_projections.xlsx", sheet_name="Data", skiprows=range(1, 107),  nrows=1)
    data3 = pd.read_excel("Data/Original/P_Data_Extract_From_Population_estimates_and_projections.xlsx", sheet_name="Data", skiprows=range(1, 149),  nrows=1)
    data4 = pd.read_excel("Data/Original/P_Data_Extract_From_Population_estimates_and_projections.xlsx", sheet_name="Data", skiprows=range(1, 114),  nrows=1)
    data5 = pd.read_excel("Data/Original/P_Data_Extract_From_Population_estimates_and_projections.xlsx", sheet_name="Data", skiprows=range(1, 2),  nrows=2)

    # Age dependency ratio, old
    # Age dependency ratio, young
    # Population ages 15-64 (% of total population)
    # Population ages 20-24, female (% of female population)
    # Population ages 65 and above (% of total population)

    data2 = data2.drop(data2.columns[[0, 1, 3]], axis=1)
    data3 = data3.drop(data3.columns[[0, 1, 3]], axis=1)
    data4 = data4.drop(data4.columns[[0, 1, 3]], axis=1)
    data5 = data5.drop(data5.columns[[0, 1, 3]], axis=1)


    data2 = data2.transpose().reset_index()
    data3 = data3.transpose().reset_index()
    data4 = data4.transpose().reset_index()
    data5 = data5.transpose().reset_index()

    new_column_names = data2.iloc[0]
    data2.columns = new_column_names
    data2 = data2.drop(0)

    new_column_names = data3.iloc[0]
    data3.columns = new_column_names
    data3 = data3.drop(0)

    new_column_names = data4.iloc[0]
    data4.columns = new_column_names
    data4 = data4.drop(0)

    new_column_names = data5.iloc[0]
    data5.columns = new_column_names
    data5 = data5.drop(0)

    data2.rename(columns={'Series Name': 'Year'}, inplace=True)
    data3.rename(columns={'Series Name': 'Year'}, inplace=True)
    data4.rename(columns={'Series Name': 'Year'}, inplace=True)
    data5.rename(columns={'Series Name': 'Year'}, inplace=True)

    merged_data = pd.merge(data2, data3, on='Year')
    merged_data = pd.merge(merged_data, data4, on='Year')
    merged_data = pd.merge(merged_data, data5, on='Year')

    merged_data['Year'] = merged_data['Year'].str.extract('(\d{4})')

    merged_data = merged_data[merged_data['Year'].astype(int) < 2022]

    return merged_data

def smoothingHolt_births():
    columns_births = ['Year', 'Net migration', 'Population in largest city', 'Population growth (annual %)', 'Population, total', 
                      'Rural population', 'Rural population (% of total population)', 'Rural population growth (annual %)', 
                      'Urban population (% of total population)', 'Urban population', 'Population in the largest city (% of urban population)',
                      'Birth rate, crude (per 1,000 people)']
    data = pd.read_excel("Data/Original/WorldBank_transformed.xlsx", sheet_name="Sheet1", usecols=columns_births)
    data = data.apply(pd.to_numeric, errors='coerce')

    return data

def smoothingHolt_deaths():
    columns_deaths = ['Year', 'Life expectancy at birth, total (years)', 'Urban population (% of total population)', 
                      'Urban population', 'Survival to age 65, female (% of cohort)', 'Survival to age 65, male (% of cohort)', 
                      'Rural population', 'Rural population (% of total population)', 'Rural population growth (annual %)', 
                      'Population, total', 'Net migration', 'Population in largest city', 'Population growth (annual %)', 
                      'Population ages 80 and above, male (% of male population)', 'Population in the largest city (% of urban population)',
                      'Death rate, crude (per 1,000 people)']
    data = pd.read_excel("Data/Original/WorldBank_transformed.xlsx", sheet_name="Sheet1", usecols=columns_deaths)
    data = data.apply(pd.to_numeric, errors='coerce')

    return data

def get_worldbankForBirths():
    columns_births = ['Year', 'Net migration', 'Population in largest city', 'Population growth (annual %)', 'Population, total', 
                      'Rural population', 'Rural population (% of total population)', 'Rural population growth (annual %)', 
                      'Urban population (% of total population)', 'Urban population', 'Population in the largest city (% of urban population)',
                      'Birth rate, crude (per 1,000 people)']
    data = pd.read_excel("Data/Original/WorldBank_transformed.xlsx", sheet_name="Sheet1", usecols=columns_births)
    data = data.apply(pd.to_numeric, errors='coerce')
    data2 = worldBank_populationsEstimates()
    data2['Year'] = data2['Year'].astype(int)

    merged_data = pd.merge(data2, data, on='Year')
    merged_data = merged_data.apply(pd.to_numeric, errors='coerce')
    return merged_data

def get_worldbankForDeaths():
    columns_deaths = ['Year', 'Life expectancy at birth, total (years)', 'Urban population (% of total population)', 
                      'Urban population', 'Survival to age 65, female (% of cohort)', 'Survival to age 65, male (% of cohort)', 
                      'Rural population', 'Rural population (% of total population)', 'Rural population growth (annual %)', 
                      'Population, total', 'Net migration', 'Population in largest city', 'Population growth (annual %)', 
                      'Population ages 80 and above, male (% of male population)', 'Population in the largest city (% of urban population)',
                      'Death rate, crude (per 1,000 people)']
    data = pd.read_excel("Data/Original/WorldBank_transformed.xlsx", sheet_name="Sheet1", usecols=columns_deaths)

    data2 = worldBank_populationsEstimates()
    data2['Year'] = data2['Year'].astype(int)

    merged_data = pd.merge(data2, data, on='Year')
    merged_data = merged_data.apply(pd.to_numeric, errors='coerce')
    return merged_data

def get_worldbankEconomics():
    data = pd.read_excel("Data/Original/P_Data_Extract_From_Global_Economic_Monitor_(GEM).xlsx", sheet_name="Data", skiprows=range(1, 5),  nrows=3)

# 'Exchange rate, new LCU per USD extended backward, period average,,',
#                 'CPI Price, seas. adj.,,,',
#                 'CPI Price,not seas.adj,,,',

    data = data.drop(data.columns[[0, 1, 3]], axis=1)
    data = data.transpose().reset_index()

    new_column_names = data.iloc[0]
    data.columns = new_column_names
    data = data.drop(0)
    data.rename(columns={'Series': 'Year'}, inplace=True)

    data_filtered = data[data['Year'].str.contains(r'^\d{4}\s\[\d{4}\]$', regex=True)]
    data_filtered['Year'] = data_filtered['Year'].str.extract('(\d{4})')

    data_filtered = data_filtered.apply(pd.to_numeric, errors='coerce')

    data_filtered = data_filtered.rename(columns={
    'Exchange rate, new LCU per USD extended backward, period average,,': 'Exchange rate',
    'CPI Price, seas. adj.,,,': 'CPI Price, seasonal',
    'CPI Price,not seas.adj,,,': 'CPI Price'
    })
    return data_filtered


#-------------------------------------------------------------------<< Economic data (not from WorldBank) >>----------------------------------------------------------------------
def leave_numbersOnly(column):
    column_cleaned = column.str.replace(r'[^0-9.]', '', regex=True)
    return column_cleaned.astype(float)

def columns_to_float(data):
    data = data.astype(float)
    data['Year'] = data['Year'].astype(int)
    return data

def get_maddisonProjectData():
    data = pd.read_excel("Data/Original/maddison-project-database.xlsx")
    return columns_to_float(data)

def get_Imports_Exports(): # Year | Imports-Billions of US $ | % of GDP-Imports | Exports-Billions of US $ | % of GDP-Exports || trade_balance || trade_ratio
    data = pd.read_excel("Data/Original/Imports_Exports.xlsx")
    data.iloc[:, 1:] = data.iloc[:, 1:].apply(leave_numbersOnly)

    data = columns_to_float(data)

    data['trade_balance'] = data['Imports-Billions of US $'] - data['Exports-Billions of US $']
    data['trade_ratio'] = data['Exports-Billions of US $'] / data['Imports-Billions of US $']

    return data

def get_inflation():
    data = pd.read_excel("Data/Original/Inflation.xlsx") # Year | Inflation Rate (%) | Inflation_Annual Change
    data.iloc[:, 1:] = data.iloc[:, 1:].apply(leave_numbersOnly)
    return columns_to_float(data)

def get_unemployment(): # Year Unemployment Rate (%) |  Unemployment_Annual Change |  % of Total Labor Force Ages 15-24  | Employment Annual Change, 15-24
    data = pd.read_excel("Data/Original/Unemployment.xlsx")
    data.iloc[:, 1:] = data.iloc[:, 1:].apply(leave_numbersOnly)

    data = data.rename(columns={
    'Annual Change, 15-24': 'Employment Annual Change, 15-24'
    })
    return columns_to_float(data)

def get_manufacturingOutput(): #  Year  |    Billions of US $  |    % of GDP
    data = pd.read_excel("Data/Original/ManufacturingOutput.xlsx")
    data.iloc[:, 1:] = data.iloc[:, 1:].apply(leave_numbersOnly)


    data = data.rename(columns={
    '% of GDP': '% of GDP-Manufacturing',
    'Billions of US $': 'Manufacturing-Billions of US $'
    })

    return columns_to_float(data)






