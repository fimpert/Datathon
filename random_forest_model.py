import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score, accuracy_score, classification_report
import matplotlib.pyplot as plt



# load excel file
df = pd.read_excel('Access_to_a_Livable_Planet_Dataset.xlsx')


df = df.fillna(0)
print(df.columns.tolist())
df.columns = df.columns.str.strip() 

# variables
features = [
    'Good Days',
    'Very Unhealthy Days',
    'Hazardous Days',
    '90th Percentile AQI',
    # 'Days CO',
    # 'Days NO2',
    'Days PM2.5',
    'Days PM10',
    'Days Ozone',
    'Median AQI'
]
x = df[features]
y = df['Max AQI']

# train model with 75/20 split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# r^2 and RMSE score calculations
y_pred = rf_model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", root_mean_squared_error(y_test, y_pred))

y_pred = rf_model.predict(X_test)
print("R2:", r2_score(y_test, y_pred))
print("RMSE:", root_mean_squared_error(y_test, y_pred))


# feature importance bar chart
importances = rf_model.feature_importances_
plt.bar(features, importances)
plt.title("Feature Importance")
plt.show()

# plot predicted vs actual
y_pred = rf_model.predict(x)
plt.scatter(y, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # 45-degree line
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Random Forest Predictions vs Actual")
plt.show()

