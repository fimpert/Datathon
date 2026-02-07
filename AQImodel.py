import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# save filepath to variable for easier access
air_file_path = 'Files/Access_to_a_Livable_Planet_Dataset.xlsx'
air_data = pd.read_excel(
    air_file_path,
)

# air_features = ['Days with AQI', '90th Percentile AQI', 'Median AQI']
X = air_data[['Days with AQI']]
y = air_data['Max AQI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Instantiate StandardScaler.
scaler = StandardScaler()

# Fit and transform training data.
X_train_scaled = scaler.fit_transform(X_train)

# Also transform test data.
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Calculate and print R^2 score.
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2:.4f}")