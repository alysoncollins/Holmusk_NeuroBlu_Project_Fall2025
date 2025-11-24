import neuroblu
import polars as pl
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
#import time
#from scipy.stats import shapiro, normaltest

query = """
WITH concept_filter AS (

 SELECT concept_id, concept_name
 FROM concept
 WHERE concept_id IN (
 SELECT DISTINCT unit_concept_id
 FROM measurement
 WHERE measurement_concept_id = 4302666

 ) 
)
select
 m.value_as_number,
 c.concept_name
from measurement m
join concept_filter c
 on m.unit_concept_id = c.concept_id
WHERE m.value_as_number IS NOT NULL

"""
### Program doesn't finish running if limit =100000000 or no limit

# Run query
values = neuroblu.get_query(query)

# Convert to Polars DataFrame
df = pl.DataFrame(values)
data = df.select(pl.col("value_as_number").drop_nulls())
X = data.to_numpy().reshape(-1, 1)


# Normality Tests
#print("\nNormality Tests:")
#stat, p = shapiro(X.ravel())
#print(f"Shapiro-Wilk Test: Statistics={stat:.3f}, p={p:.3f}")
#stat, p = normaltest(X.ravel())
#print(f"D’Agostino’s K² Test: Statistics={stat:.3f}, p={p:.3f}")


# Local Outlier Factor
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = clf.fit_predict(X)
X_scores = clf.negative_outlier_factor_

# Add predictions to Polars DataFrame
df = df.with_columns([
    pl.Series("lof_score", X_scores),
    pl.Series("is_outlier", (y_pred == -1).astype(int))
])

# Summary 
num_outliers = df["is_outlier"].sum()
print(f"Detected {num_outliers} potential outliers out of {len(df)} samples.\n")

# Save outliers to CSV 
outlier_file = "outliers.csv"
df.filter(pl.col("is_outlier") == 1).write_csv(outlier_file)

# Graphing 
plt.figure(figsize=(10, 6))
plt.scatter(range(len(X)), X, c=y_pred, cmap="coolwarm_r", s=8)
plt.title("Local Outlier Factor (LOF) Outlier Detection")
plt.xlabel("Sample Index")
plt.ylabel("Temperature Value")
plt.colorbar(label="Outlier (1=normal, -1=outlier)")
plt.tight_layout()
plt.savefig("lof_outlier_scatter.png", dpi=300)
plt.close()

