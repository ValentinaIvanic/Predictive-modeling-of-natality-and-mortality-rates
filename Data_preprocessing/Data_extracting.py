import pandas as pd

def get_dataFromEurostat(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, skiprows=7)

    filtered_df = df[df[0].isin(['TIME', 'Croatia'])]

    filtered_df = filtered_df[~filtered_df[0].isna()]
    filtered_df = filtered_df.dropna(axis=1, how='all')
    return filtered_df

def get_dataFromHNB(file_path, sheet_name, header):
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=header, skiprows=1)

    df.columns = df.iloc[0]
    df = df.drop(0)

    df.reset_index(drop=True, inplace=True)
    filtered_df = df.dropna(how='all')

    
    # print(filtered_df)

    return filtered_df

# --------------------------------------------<< Data from Eurostat >> ----------------------------------------------------


def get_deaths():
    return get_dataFromEurostat("Data\Deaths (total) by month.xlsx", "Sheet 1")

def get_births():
    return get_dataFromEurostat("Data\Live births (total) by month.xlsx", "Sheet 1")

def get_population():
    df = get_dataFromEurostat("Data\Population on 1 January by age and sex.xlsx", "Sheet 1")
    data_transposed = df.T
    data_transposed.columns = data_transposed.iloc[0]
    data_transposed = data_transposed[1:].reset_index(drop=True)
    data_transposed = data_transposed.dropna(subset=['TIME'], axis=0)

    data_transposed['TIME'] = data_transposed['TIME'].astype(int)
    data_transposed['Croatia'] = data_transposed['Croatia'].astype(int)

    data_transposed.rename(columns={'TIME': 'Year', 'Croatia': 'Population'}, inplace=True)


    return data_transposed
#-------

def get_womenAgeFirstBirth():
    return get_dataFromEurostat("Data\Fertility indicators - mean women age at first child birth.xlsx", "Sheet 1")

def get_womenCompletedAgeFirstMarriage():
    return get_dataFromEurostat("Data\First marriage rates by age and sex - females.xlsx", "Sheet 1")

def get_manCompletedAgeFirstMarriage():
    return get_dataFromEurostat("Data\First marriage rates by age and sex - males.xlsx", "Sheet 1")

def get_womenAgeFirstMarriage():
    return get_dataFromEurostat("Data\First-time marrying persons by age and sex.xlsx", "Sheet 1")

def get_HICPbyMonth():
    return get_dataFromEurostat("Data\Harmonised index of consumer prices mjesečno.xlsx", "Sheet 1")

# --------------------------------------------<</ Data from Eurostat >> ----------------------------------------------------



# --------------------------------------------<< Data from HNB >> ----------------------------------------------------

def get_PriceIndexResidentalBuilding():
    df = get_dataFromHNB("Data\Indeksi cijena stambenih objekata.xlsx", "HRV", 2)
    df = df.iloc[:-4]
    print(df)
    return df

def get_ConsumerIndexes():
    df = get_dataFromHNB("Data\Indeksi pouzdanja, očekivanja, raspoloženja potrošača.xlsx", "HRV", 3)
    df = df.iloc[:-1]
    print(df)
    return df

def get_ConsumerProducerPricesIndex():
    df = get_dataFromHNB("Data\Indeksi potrošačkih cijena i prozivođačkih cijena industrije.xlsx", "HRV", 2)
    df = df.iloc[:-3]
    print(df)
    return df

def get_FundamentalConsumerPriceIndexes():
    df = get_dataFromHNB("Data\Temeljni indeksi potrošačkih cijena.xlsx", "HRV", 4)
    df = df.iloc[:-2]
    print(df)
    return df

def get_HICP():
    df = get_dataFromHNB("Data\Harmonizirani indeksi potrošačkih cijena.xlsx", "HRV", 2)
    df = df.iloc[:-1]
    print(df)
    return df

def get_AverageSalaryByMonth():
    df = get_dataFromHNB("Data\Prosječna mjesečna neto plaća i indeksi.xlsx", "EUR", 3)
    df = df.iloc[:-2]
    print(df)
    return df

# --------------------------------------------<</ Data from HNB >> ----------------------------------------------------


# get_births()
# get_womenAgeFirstBirth()
# get_womenCompletedAgeFirstMarriage()
# get_manCompletedAgeFirstMarriage()
# get_HICPbyMonth()
# get_births()
get_population()

# get_PriceIndexResidentalBuilding()
# get_ConsumerIndexes()
# get_ConsumerProducerPricesIndex()
# get_FundamentalConsumerPriceIndexes()
# get_HICP()
# get_AverageSalaryByMonth()