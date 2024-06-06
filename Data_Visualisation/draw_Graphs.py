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

    print("Years:", years)
    print("Original data:", original_data)
    print("Model predictions:", model_predictions)

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