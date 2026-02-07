import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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


# pull relevant values
X = df[['Percentage of Unhealthy Days', 'Max AQI']].values

# scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# k means
k = 6
kmeans_model = KMeans(n_clusters=k, init='k-means++', random_state=42)
df["cluster"] = kmeans_model.fit_predict(X_scaled)


counties = gpd.read_file("cb_2018_us_county_500k.shp")
df["fips"] = df["fips"].astype(str).str.zfill(5)
counties["GEOID"] = counties["GEOID"].astype(str).str.zfill(5)


print(df[["State", "County", "fips", "cluster"]].head())
print(df[["State", "County", "fips", "cluster"]].tail())
print(df["cluster"].unique())

# Suppose counties shapefile has GEOID column
counties = counties.merge(df[['fips', 'cluster']], left_on='GEOID', right_on='fips', how='left')

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
counties.plot(column='cluster', cmap='tab10', legend=True, linewidth=0.5, edgecolor='black', ax=ax)
ax.set_title("K-means Clusters of Counties")
ax.axis('off')
plt.show()


# print(df["fips"].head())
# print(df["fips"].dtype)

# print(counties["GEOID"].head())
# print(counties["GEOID"].dtype)