from __imports import *
from factor_analyzer import FactorAnalyzer 

merged_data = get_dataBirths()

top_features = ['Population ages 65 and above (% of total population)', 
                'Age dependency ratio, old', 
                'Urban population (% of total population)', 
                'Rural population (% of total population)', 
                '% of GDP-Exports', 'CPI Price', 
                'CPI Price, seasonal', 'Age dependency ratio, young', 
                'GDP_per_capita_2011_prices', 
                'Population ages 15-64 (% of total population)']

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(merged_data[top_features])

factor_model = FactorAnalyzer(n_factors=3, rotation='varimax') 
factor_model.fit(scaled_data) 

factor_loadings = factor_model.loadings_
print("Faktorski utezi:")
print(factor_loadings)

explained_variance = factor_model.get_factor_variance()
print("Objašnjena varijanca po faktoru:")
print(explained_variance)

eigenvalues = factor_model.get_eigenvalues()

plt.scatter(range(1, len(eigenvalues[0]) + 1), eigenvalues[0])
plt.plot(range(1, len(eigenvalues[0]) + 1), eigenvalues[0], marker='o', linestyle='-')
plt.title('Scree Plot')
plt.xlabel('Faktori')
plt.ylabel('Eigenvalues')
plt.grid()
plt.show()