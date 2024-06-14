from __imports import *

merged_data = get_dataBirths()
features = merged_data.columns.tolist()

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(merged_data)
merged_data_scaled = pd.DataFrame(scaled_data, columns=merged_data.columns) 


print(merged_data_scaled['Births'])

correlations = {}
for feature in features:
    correlation = merged_data_scaled[[feature, 'Births']].corr().iloc[0, 1]
    if feature != "Births":
        correlations[feature] = correlation

sorted_correlations = sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True)


print("Correlations:")
for feature, correlation in sorted_correlations:
    print(f"{feature:60} : {correlation:.4f}")

correlation_df = pd.DataFrame(sorted_correlations, columns=['Feature', 'Correlation'])

plt.figure(figsize=(12, 10))
sns.barplot(data=correlation_df, x='Correlation', y='Feature', palette='viridis')
plt.title('Korelacija varijabli s brojem rođenih', fontsize=17)
plt.xlabel('Koeficijent korelacije', fontsize=15)
plt.ylabel('Imena varijabli', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=14) 
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(merged_data['Year'], merged_data_scaled['Births'], label='Births')
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Time Series of Births')
plt.legend()
plt.show()


top_features = [item[0] for item in sorted_correlations[:5]]
# top_features.remove('Year')
top_features.append('Births')

print(top_features)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(merged_data[top_features])

scaled_df = pd.DataFrame(scaled_data, columns=top_features)
scaled_df['Year'] = merged_data['Year']

plt.figure(figsize=(14, 10))
plt.plot(scaled_df['Year'], scaled_df['Births'], label='Births', linewidth = 6, color='red')

top_features.remove('Births')

for feature in top_features:
    plt.plot(scaled_df['Year'], scaled_df[feature], label=feature, linewidth = 3)

plt.xlabel('Godina', fontsize=15)
plt.ylabel('Skalirane vrijednosti varijabli', fontsize=15)
plt.title('5 najviše koreliranih varijabli s brojem rođenih(skalirano)', fontsize=16)
plt.legend(fontsize=14)
plt.show()