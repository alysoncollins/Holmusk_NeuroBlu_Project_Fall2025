import neuroblu
import polars as pl
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy.stats import shapiro, normaltest
from sklearn.preprocessing import StandardScaler, LabelEncoder


def main():
     # select which measurement based on index
    selector = 0
    # ['measurement', 'concept_id', 'lower', 'upper']
    measurements = [
        ['temperature', 4302666, 95, 100.4],
        ['body mass index', 4245997, 17, 29.9],
        ['diastolic blood pressure', 4154790, 50, 89],
        ['systolic blood pressure', 4152194, 80, 139],
        ['body weight', 4099154, 0, 1000],
        ['body height measure', 4177340, 0, 250],
        ['pulse rate', 4301868, 40, 120],
        ['pulse oximetry', 4098046, 0.85, 1],
        ['fasting glucose', 3037110, 50, 125],
        ['A1c', 37392407, 0.035, 0.064],
    ]

    measurement, concept_id, lower, upper = measurements[selector]

    

    query = f"""
    WITH concept_filter AS (
        SELECT concept_id, concept_name
        FROM concept
        WHERE concept_id IN (
            SELECT DISTINCT unit_concept_id
            FROM measurement
            WHERE measurement_concept_id = {concept_id}
        )
    )
    SELECT DISTINCT
        m.value_as_number,
        sex_concept.concept_name AS sex,
        EXTRACT(YEAR FROM m.measurement_date) - p.year_of_birth AS age,
        c.concept_name AS unit_name,
        p.person_id
        m.measurement_date,
    FROM measurement m
    JOIN concept_filter c
        ON m.unit_concept_id = c.concept_id
    JOIN person p
        ON m.person_id = p.person_id
    LEFT JOIN concept AS sex_concept
        ON p.gender_concept_id = sex_concept.concept_id
    WHERE m.value_as_number IS NOT NULL;
    """
    
     # Run the query
    values = neuroblu.get_query(query)
    df = pl.DataFrame(values)

    # --- Average multiple same-day measurements per person ---
    if "person_id" in df.columns and "measurement_date" in df.columns:
        df = (
            df.group_by(["person_id", "measurement_date"])
            .agg([
                pl.col("value_as_number").mean().alias("value_as_number"),
                pl.col("sex").first(),
                pl.col("age").first(),
                pl.col("unit_name").first(),
            ])
        )
        print(f"Averaged same-day duplicates: {len(df)} unique (person, date) pairs remaining.")
    else:
        print("Warning: person_id or measurement_date missing; skipping daily averaging.")

    # Filter out-of-range values
    out_of_range = df.filter((pl.col("value_as_number") < lower) | (pl.col("value_as_number") > upper))
    in_range = df.filter((pl.col("value_as_number") >= lower) & (pl.col("value_as_number") <= upper))

    # Save out-of-range values to CSV
    out_of_csv = f"{measurement}_out_of_range.csv"
    out_of_range.write_csv(out_of_csv)
    print(f"Saved {len(out_of_range)} out-of-range rows to {out_of_csv}")

    # (keep only value_as_number, age, sex)
    in_range = in_range.drop_nulls(subset=["value_as_number", "age", "sex"])

    # Encode sex as numeric (e.g., male/female → 0/1)
    encoder = LabelEncoder()
    in_range = in_range.with_columns(
        pl.Series("sex_encoded", encoder.fit_transform(in_range["sex"].to_list()))
    )

    # Select features
    X = in_range.select(["value_as_number", "age", "sex_encoded"])

    # Standardize (flatten) to avoid one feature dominating
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.to_numpy())

    # Run model
    df_results, anomalies, scores = isolation_forest(in_range, X_scaled)

    # Graphs
    graph_distribution(measurement, df_results, anomalies)
    graph_anomalies(measurement, scores)



def isolation_forest(df, X_scaled):
    iso = IsolationForest(contamination=0.02, random_state=42)
    y_pred = iso.fit_predict(X_scaled)
    scores = iso.decision_function(X_scaled)

    # Add results to df
    df = df.clone()
    df = df.with_columns([
        pl.Series("anomaly_label", y_pred),
        pl.Series("anomaly_score", scores),
    ])

    anomalies = df.filter(pl.col("anomaly_label") == -1)
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

if __name__ == "__main__":
    main()
