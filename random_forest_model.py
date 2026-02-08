import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score, accuracy_score, classification_report
import matplotlib.pyplot as plt



# load excel file
df = pd.read_excel('Access_to_a_Livable_Planet_Dataset.xlsx')
crosswalk = pd.read_csv("ssa_fips_state_county_2025.csv")

# data cleaning
# cleaning normalizing dataset county and state names
df["County"] = (df["County"].str.lower().str.replace(" County", "", regex=False).str.strip())
df["State"] = (df["State"].str.lower().str.replace(" State", "", regex=False).str.strip())

# renaming crosswalk columns and normalizing county and state names
crosswalk = crosswalk.rename(columns={"countyname_fips": "County","state_name": "State"})
crosswalk["County"] = (crosswalk["County"].str.lower().str.replace(" County", "", regex=False).str.strip())
crosswalk["State"] = ( crosswalk["State"] .str.lower().str.replace(" State", "", regex=False).str.strip())

#print(df.columns)
#print(crosswalk.columns)

# merging datasets
df = df.merge(
    crosswalk,
    left_on=["State", "County"],
    right_on=["State", "County"],
    how="left"
)


df = df.fillna(0)
df = df.rename(columns={"fipscounty": "fips"})
df['fips'] = df['fips'].astype(str).str.split('.').str[0].str.zfill(5)
print(df.columns.tolist())
df.columns = df.columns.str.strip() 

features = [
    'Very Unhealthy Days',
    'Hazardous Days',
    '90th Percentile AQI'
]
x = df[features]
y = df['Max AQI']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=17)


rf_model = RandomForestRegressor(n_estimators=100, random_state=17)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", root_mean_squared_error(y_test, y_pred))


importances = rf_model.feature_importances_
plt.bar(features, importances)
plt.title("Feature Importance")
plt.show()


y_pred = rf_model.predict(X_test)
print("R2:", r2_score(y_test, y_pred))
print("RMSE:", root_mean_squared_error(y_test, y_pred))