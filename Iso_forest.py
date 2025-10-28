import neuroblu
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy.stats import shapiro, normaltest


def main():
    ###should have some kind of list with measurements and queries and whatnot
    measurement = "Pulse_Oximetry"

    query = """
    SELECT 
        m.value_as_number,
        c.concept_name AS unit_name
    FROM measurement m
    JOIN concept c
        ON m.unit_concept_id = c.concept_id
    WHERE m.measurement_concept_id = 4098046
    LIMIT 1000
    """
    values = neuroblu.get_query(query)
    df = pd.DataFrame(values)
    data = df["value_as_number"].dropna()
    df_results, anomalies, scores = isolation_forest(df, data)
    graph_distribution(measurement, df_results, anomalies)
    graph_anomalies(measurement, scores)


def isolation_forest(df,data_if):
    # Reshape data for model input
    x = data_if.values.reshape(-1, 1)

    # Initialize model
    iso = IsolationForest(contamination=0.02, random_state=42) # Expect ~2% of points to be anomalies (can adjust if needed)
    y_pred = iso.fit_predict(x)
    scores = iso.decision_function(x)  # anomaly scores (lower = more anomalous)

    # Add results to df
    df = df.loc[data_if.index].copy()
    df["anomaly_label"] = y_pred
    df["anomaly_score"] = scores

    # Extract and display anomalies
    anomalies = df[df["anomaly_label"] == -1]
    print(f"Detected {len(anomalies)} anomalies out of {len(df)} total samples.\n")

    print("Example anomalies:")
    print(anomalies.head())

    return df, anomalies, scores

def graph_distribution(measurement, df, anomalies):
    #graph of anomalies
    plt.figure(figsize=(8,5))
    plt.hist(df["value_as_number"], bins=30, color="lightgrey", edgecolor="black", label="Normal Data")
    plt.scatter(anomalies["value_as_number"], [0]*len(anomalies), color="red", label="Anomalies")
    plt.title("Anomaly Detection (Isolation Forest - Full Dataset)")
    plt.xlabel(measurement)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{measurement}_anomalies_full.png", dpi=300)
    plt.close()

def graph_anomalies(measurement, scores):
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50, color="lightblue", edgecolor="black")
    plt.axvline(x=np.percentile(scores, 14.7), color="red", linestyle="--", label="Anomaly Threshold")
    plt.title(f"{measurement}Isolation Forest Decision Score Distribution")
    plt.xlabel("Decision Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{measurement}Decision_Score_Distribution.png", dpi=300)
    plt.close()

def normality(local_data):
    #not currently called
    # Visualize distribution
    plt.figure(figsize=(8,5))
    plt.hist(local_data, bins=30, edgecolor="black", color="skyblue")
    plt.title("Pulse Oximetry Values (Full Dataset)")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("pulse_oximetry_hist_full.png", dpi=300)
    plt.show()
    # Normality tests
    print("\nNormality Tests:")
    stat, p = shapiro(local_data)
    print(f"Shapiro-Wilk Test: Statistics={stat:.3f}, p={p:.3f}")
    stat, p = normaltest(local_data)
    print(f"D’Agostino’s K² Test: Statistics={stat:.3f}, p={p:.3f}")
    plt.close()

main()
