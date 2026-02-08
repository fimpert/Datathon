import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from shapely.affinity import scale, translate
from sklearn.metrics import silhouette_score
from pandas.api.types import CategoricalDtype



# load excel file
df = pd.read_excel('Access_to_a_Livable_Planet_Dataset.xlsx')
crosswalk = pd.read_csv("ssa_fips_state_county_2025.csv")

# data cleaning
# cleaning normalizing dataset county and state names
df["County"] = (df["County"].str.lower().str.replace(" County", "", regex=False).str.strip())
df["State"] = (df["State"].str.lower().str.replace(" State", "", regex=False).str.strip())

# renaming crosswalk columns and normalizing county and state names
crosswalk = crosswalk.rename(columns={"countyname_fips": "County","state_name": "State", "fipscounty": "fips"})
crosswalk["County"] = (crosswalk["County"].str.lower().str.replace(" County", "", regex=False).str.strip())
crosswalk["State"] = ( crosswalk["State"] .str.lower().str.replace(" State", "", regex=False).str.strip())

#print(df.columns)
#print(crosswalk.columns)

# merging dataset for fips code 
df = df.merge(
    crosswalk,
    left_on=["State", "County"],
    right_on=["State", "County"],
    how="left"
)


df = df.fillna(0)
df['fips'] = df['fips'].astype(str).str.split('.').str[0].str.zfill(5)


# pull relevant values
X = df[[
    'Very Unhealthy Days',
    'Hazardous Days',
    '90th Percentile AQI']].values

# scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# k means
k = 4
kmeans_model = KMeans(n_clusters=k, init='k-means++', random_state=42)
df["cluster"] = kmeans_model.fit_predict(X_scaled)

# open US county shapefile
counties = gpd.read_file("cb_2018_us_county_500k.shp")
counties["GEOID"] = counties["GEOID"].astype(str).str.zfill(5)

# remove carrebiean islands
counties = counties[~counties['STATEFP'].isin(['60','66','69','72','78'])]


# print(df[["State", "County", "fips", "cluster"]].head())
# print(df[["State", "County", "fips", "cluster"]].tail())
# print(df["cluster"].unique())


# Sort clusters by mean % unhealthy days and reorder from lowest to highest
cluster_summary = df.groupby('cluster')['90th Percentile AQI'].mean()
ordered_clusters = cluster_summary.sort_values(ascending=True).index.tolist()
cluster_order_map = {old: new for new, old in enumerate(ordered_clusters)}
df['cluster_ordered'] = df['cluster'].map(cluster_order_map)


# label clusters
labels = {
    0: "Low Pollution",
    1: "Moderate Pollution",
    2: "High Pollution",
    3: "Severe Pollution"
}
df['cluster_label'] = df['cluster_ordered'].map(labels)

# merge and plot
counties = counties.merge(df[['fips', 'cluster_label']], left_on='GEOID', right_on='fips', how='left')
counties['cluster_label'] = counties['cluster_label'].astype(CategoricalDtype(categories=["Low Pollution", "Moderate Pollution", "High Pollution", "Severe Pollution"], ordered=True))


print(df.groupby('cluster_ordered')['90th Percentile AQI'].mean())
print(df.groupby('cluster_label')['90th Percentile AQI'].mean())

# FORMATTING
counties = counties.to_crs("EPSG:5070")
alaska = counties[counties['STATEFP'] == '02'].copy()
hawaii = counties[counties['STATEFP'] == '15'].copy()
conus = counties[(counties['STATEFP'] != '02') & (counties['STATEFP'] != '15')].copy()
alaska['geometry'] = alaska['geometry'].apply(lambda x: scale(x, xfact=0.5, yfact=0.5, origin=(0, 0)))
alaska['geometry'] = alaska['geometry'].apply(lambda x: translate(x, xoff=-2_000_000, yoff=0))
hawaii['geometry'] = hawaii['geometry'].apply(lambda x: translate(x, xoff=3_000_000, yoff = -1_000_000))
counties = pd.concat([conus, alaska, hawaii])
counties['geometry'] = counties['geometry'].apply(lambda x: scale(x, xfact=1.5, yfact=1.5, origin=(0, 0)))

fig, ax = plt.subplots(1, 1, figsize=(15, 10), dpi=120)
counties.plot(
    column='cluster_label', 
    cmap='Reds', 
    categorical=True,
    legend=True, 
    linewidth=0.5, 
    # edgecolor='black', 
    ax=ax, 
    legend_kwds={"title": "Air Quality Cluster", "loc": "lower left", "fontsize": 10}
)
counties.boundary.plot(
    ax=ax,
    color='black',
    linewidth=0.5
)
# ax.set_xlim(-180, -60)
# ax.set_ylim(15, 75)

ax.set_title("County-level Air Quality Clusters (2025)")
ax.axis('off')
plt.show()



# elbow method
inertias = []
K = range(1, 11)

for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.plot(K, inertias, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()

# silhouette score
sil_scores = {}

for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    sil_scores[k] = score
    print(f"k={k}, silhouette score={score:.3f}")

# print(df["fips"].head())
# print(df["fips"].dtype)

# print(counties["GEOID"].head())
# print(counties["GEOID"].dtype)