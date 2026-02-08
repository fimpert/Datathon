import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mplcursors
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor


# save filepath to variable for easier access
air_file_path = 'Files/Access_to_a_Livable_Planet_Dataset.xlsx'
air_data = pd.read_excel(
    air_file_path,
)

# SET-UP
# Figure out which test, 
multiTest = True;
accuracyTest = ['']
air_features = ['Very Unhealthy Days', 'Hazardous Days', 'Percentage of Unhealthy Days']
# air_features = ['Percentage of Unhealthy Days']

for variable in air_features: 
    air_data['log_' + variable] = np.log1p(air_data[variable]) 

X = pd.DataFrame(air_data[air_features])
# X = air_data[['Percentage of Unhealthy Days']]
y = air_data['Max AQI']

# Training Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
# Fit and transform training data.
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
rmse_scores = -cross_val_score(
    model,
    X,
    y,
    cv=kf,
    scoring="neg_root_mean_squared_error"
)

# Calculate and model effectiveness
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2:.4f}")

mse = mean_squared_error(y_test, y_pred)
print(f"Mean squared error: {mse:.4f}")

rmse = mse ** 0.5
print(f"Root mean squared error: {rmse:.4f}") 

print("Mean RMSE:", rmse_scores.mean())

print("Std RMSE:", rmse_scores.std())

if (multiTest):
    # Compute Variance Inflation Factor (VIF) for each feature.
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]


    # Print VIF values.
    print("\nVariance Inflation Factor (VIF) for each feature:\n", vif_data)

    print("Intercept:", model.intercept_)

    coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
    print("\nFeature Coefficients:\n", coef_df)

    # Sort dataframe by coefficients.
    coef_df_sorted = coef_df.sort_values(by="Coefficient", ascending=False)


    # Create plot.
    plt.figure(figsize=(8,6))
    plt.barh(coef_df["Feature"], coef_df_sorted["Coefficient"], color="blue")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.title("Feature Importance (Linear Regression Coefficients)")
    plt.show()

# Compute residuals.
residuals = y_test - y_pred


# Create plots.
fig = plt.figure(figsize=(12,5))

# Plot 1: Residuals Distribution.
plt.subplot(1,2,1)
sns.histplot(residuals, bins=30, kde=True, color="blue")
plt.axvline(x=0, color='red', linestyle='--')
plt.title("Residuals Distribution")
plt.xlabel("Residuals (y_actual - y_predicted)")
plt.ylabel("Frequency")


# Plot 2: Regression Fit
plt.subplot(1,2,2)
scatter = sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
# Actual Fit
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Perfect fit line
plt.title("Regression Fit")
plt.xlabel("Actual Max AQI")
plt.ylabel("Predicted Max AQI")
cursor = mplcursors.cursor(scatter, hover=True)

def on_add(sel):
    x_val, y_val = sel.target
    dist = ((y_test - x_val)**2 + (y_pred - y_val)**2).values
    nearest_idx = dist.argmin()
    sel.annotation.set_text(air_data.iloc[nearest_idx]["County"])

cursor.connect("add", on_add)

# Show plots.
plt.tight_layout()
plt.show()