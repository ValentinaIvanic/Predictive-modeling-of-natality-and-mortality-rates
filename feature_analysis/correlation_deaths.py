from __imports import *

merged_data = get_dataDeaths()
features = merged_data.columns.tolist()


scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(merged_data)
merged_data_scaled = pd.DataFrame(scaled_data, columns=merged_data.columns) 

correlations = {}
for feature in features:
    correlation = merged_data_scaled[[feature, 'Deaths']].corr().iloc[0, 1]
    if feature != "Deaths":
        correlations[feature] = correlation

sorted_correlations = sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True)


print("Correlations:")
for feature, correlation in sorted_correlations:
    print(f"{feature:60} : {correlation:.4f}")

correlation_df = pd.DataFrame(sorted_correlations, columns=['Feature', 'Correlation'])

plt.figure(figsize=(12, 10))
sns.barplot(data=correlation_df, x='Correlation', y='Feature', palette='viridis')
plt.title('Korelacija varijabli s brojem umrlih', fontsize=17)
plt.xlabel('Koeficijent korelacije', fontsize=15)
plt.ylabel('Imena varijabli', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=14) 
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(merged_data['Year'], merged_data_scaled['Deaths'], label='Deaths')
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Time Series of Deaths')
plt.legend()
plt.show()


top_features = [item[0] for item in sorted_correlations[:7]]
top_features.remove('Year')
top_features.append('Deaths')

print(top_features)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(merged_data[top_features])

scaled_df = pd.DataFrame(scaled_data, columns=top_features)
scaled_df['Year'] = merged_data['Year']

plt.figure(figsize=(14, 10))
plt.plot(scaled_df['Year'], scaled_df['Deaths'], label='Deaths', linewidth = 6, color='red')

top_features.remove('Deaths')

for feature in top_features:
    plt.plot(scaled_df['Year'], scaled_df[feature], label=feature, linewidth = 3)

plt.xlabel('Godina', fontsize=16)
plt.ylabel('Skalirane vrijednosti varijabli', fontsize=16)
plt.title('6 najviše koreliranih varijabli s brojem umrlih(skalirano)', fontsize=16)
plt.legend(fontsize=15)
plt.show()