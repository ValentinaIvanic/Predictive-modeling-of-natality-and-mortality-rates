import matplotlib.pyplot as plt

def deviations(original_data, model_predictions, naslov):
    plt.figure()
    plt.scatter(original_data, model_predictions)
    plt.plot([min(original_data), max(original_data)], [min(original_data), max(original_data)], 'r--', lw=2)
    plt.xlabel('Stvarne vrijednosti')
    plt.ylabel('Prediktivne vrijednosti')
    plt.title(f'Usporedba stvarnih i prediktivnih vrijednosti za {naslov} set')
    plt.legend()
    plt.show()

def byYear_2datasets(years, original_data, model_predictions, naslov):

    # print("Years:", years)
    # print("Original data:", original_data)
    # print("Model predictions:", model_predictions)

    sorted_data = sorted(zip(years, original_data, model_predictions), key=lambda x: x[0])
    sorted_years, sorted_original_data, sorted_model_predictions = zip(*sorted_data)
    
    plt.figure()
    plt.plot(sorted_years, sorted_original_data, label='Stvarne vrijednosti', color='green', linewidth=2)
    plt.plot(sorted_years, sorted_model_predictions, label='Prediktivne vrijednosti',  color='red')
    plt.xlabel('Godina')
    plt.ylabel('Broj rođenih')
    plt.title(f'Usporedba stvarnih i prediktivnih podataka po godinama za {naslov} set')
    plt.legend()
    plt.show()

def train_test_testPred(years, y_train, y_test, y_predictions, train_size):
    plt.figure(figsize=(12, 6))
    
    years_train = years[:train_size]
    years_test = years[train_size:]

    plt.plot(years_train, y_train, label='Train Data', color='blue')
    plt.plot(years_test, y_test, label='Test Data', color='green')
    plt.plot(years_test, y_predictions, label='Predictions', color='red', linestyle='dashed')
    
    plt.xlabel('Year')
    plt.ylabel('Values')
    plt.title('Time Series Data and Predictions')
    plt.legend()
    plt.grid(True)
    plt.show()

def train_test_trainpred_testPred(years, y_train, y_test, y_predictions, train_predictions, train_size):
    plt.figure(figsize=(12, 6))
    
    years_train = years[:train_size]
    years_test = years[train_size:]
    
    plt.plot(years_train, y_train, label='Train Data', color='blue')
    plt.plot(years_test, y_test, label='Test Data', color='green')
    plt.plot(years_test, y_predictions, label='Y Predictions', color='red', linestyle='dashed')
    plt.plot(years_train, train_predictions, label='Train Predictions', color='purple', linestyle='dotted')
    
    plt.xlabel('Year')
    plt.ylabel('Values')
    plt.title('Time Series Data and Predictions')
    plt.legend()
    plt.grid(True)
    plt.show()

def variable_by_years(years, variable, naziv_varijable):
    plt.figure(figsize=(12, 6))
    
    plt.plot(years, variable, marker='o', linestyle='-', color='red', label='Deaths')
    
    plt.xlabel('Year')
    plt.ylabel(f'Number of {naziv_varijable}')
    plt.title(f'Number of {naziv_varijable} by Year')
    plt.legend()
    plt.grid(True)
    plt.show()