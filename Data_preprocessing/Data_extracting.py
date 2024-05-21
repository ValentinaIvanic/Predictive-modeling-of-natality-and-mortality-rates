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

def get_womenAgeFirstMarriage():
    return get_dataFromEurostat("Data\Original\First-time marrying persons by age and sex.xlsx", "Sheet 1")

def get_HICPbyMonth():
    return get_dataFromEurostat("Data\Original\Harmonised index of consumer prices mjesečno.xlsx", "Sheet 1")

# --------------------------------------------<</ Data from Eurostat >> ----------------------------------------------------



# --------------------------------------------<< Data from HNB >> ----------------------------------------------------

def get_PriceIndexResidentalBuilding():
    df = get_dataFromHNB("Data\Original\Indeksi cijena stambenih objekata.xlsx", "HRV", 2)
    df = df.iloc[:-4]
    print(df)
    return df

def get_ConsumerIndexes():
    df = get_dataFromHNB("Data\Original\Indeksi pouzdanja, očekivanja, raspoloženja potrošača.xlsx", "HRV", 3)
    df = df.iloc[:-1]
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
    df = get_dataFromHNB("Data\Original\Prosječna mjesečna neto plaća i indeksi.xlsx", "EUR", 3)
    df = df.iloc[:-2]
    print(df)
    return df

# --------------------------------------------<</ Data from HNB >> ----------------------------------------------------


# print(get_births())
# print("--------------------------------------------------------")
# get_womenAgeFirstBirth()
# get_womenCompletedAgeFirstMarriage()
# get_manCompletedAgeFirstMarriage()
# get_HICPbyMonth()
# print(get_deaths())
# print("--------------------------------------------------------")
# print(get_population())
# print("--------------------------------------------------------")


# get_PriceIndexResidentalBuilding()
# get_ConsumerIndexes()
# get_ConsumerProducerPricesIndex()
# get_FundamentalConsumerPriceIndexes()
# get_HICP()
# get_AverageSalaryByMonth()



#------------------------------------------------- << New predicted data >> -------------------------------------------

def get_predictedBirths():
    data = pd.read_csv("Data/Predicted/births_predicted.csv")
    return data

def get_predictedDeaths():
    data = pd.read_csv("Data/Predicted/deaths_predicted.csv")
    return data

