from __imports import *

merged_data = get_data()
features = merged_data.columns.tolist()

correlations = {}
for feature in features:
    correlation = merged_data[[feature, 'Births']].corr().iloc[0, 1]
    if feature != "Births":
        correlations[feature] = correlation

sorted_correlations = sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True)


print("Correlations:")
for feature, correlation in sorted_correlations:
    print(f"{feature:60} : {correlation:.4f}")

correlation_df = pd.DataFrame(sorted_correlations, columns=['Feature', 'Correlation'])

plt.figure(figsize=(12, 10))
sns.barplot(data=correlation_df, x='Correlation', y='Feature', palette='viridis')
plt.title('Correlation of Features with Birth Rate')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Features')
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(merged_data['Year'], merged_data['Births'], label='Births')
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Time Series of Births')
plt.legend()
plt.show()


top_features = [item[0] for item in sorted_correlations[:11]]
top_features.remove('Year')
top_features.append('Births')

print(top_features)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(merged_data[top_features])

scaled_df = pd.DataFrame(scaled_data, columns=top_features)
scaled_df['Year'] = merged_data['Year']

plt.figure(figsize=(14, 10))
plt.plot(scaled_df['Year'], scaled_df['Births'], label='Births', linewidth = 6)

top_features.remove('Births')

for feature in top_features:
    plt.plot(scaled_df['Year'], scaled_df[feature], label=feature)

plt.xlabel('Year')
plt.ylabel('Scaled Value')
plt.title('Top 10 Features with Highest Correlation to Births (Scaled)')
plt.legend()
plt.show()