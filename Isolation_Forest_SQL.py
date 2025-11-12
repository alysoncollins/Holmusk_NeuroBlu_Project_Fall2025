import neuroblu
import polars as pl
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def main():
     # select which measurement based on index
    selector = 0
    # ['measurement', 'concept_id', 'lower', 'upper']
    measurements = [
        ['temperature', (4302666,), 95, 100.4], #0
        ['body_mass_index', (4245997,), 17, 29.9], #1
        ['diastolic_blood_pressure', (4154790,), 50, 89], #2
        ['systolic_blood_pressure', (4152194,), 80, 139], #3
        ['pulse_rate', (4301868,), 40, 120], #4
        ['pulse_oximetry', (4098046,), 0.85, 1], #5
        ['fasting_glucose', (3037110,), 50, 125], #6
        ['A1c', (3003309, 3004410, 3005673, 3007263, 3034639, 36032094, 40762352, 42869630), 0.035, 0.064], #7
        #['body_weight', (4099154,), 0, 1000],
        #['body_height', (4177340,), 0, 275],
    ]

    measurement, concept_id, lower, upper = measurements[selector]

    

    query = f"""
    WITH
unit_concept_filter AS (
    SELECT DISTINCT c.concept_id, c.concept_name
    FROM concept c
    INNER JOIN measurement m
        ON c.concept_id = m.unit_concept_id
    WHERE m.measurement_concept_id IN {concept_id}
),
gender_concept_filter AS (
    SELECT concept_id, concept_name
    FROM concept
    WHERE concept_id IN (8507, 8532)
),
latest_bmi AS (
    SELECT
        m.person_id,
        FIRST(m.value_as_number ORDER BY m.measurement_date DESC) AS bmi
    FROM measurement m
    WHERE m.measurement_concept_id = 4245997
      AND m.value_as_number IS NOT NULL
    GROUP BY m.person_id
),
deduped AS (
    SELECT
        ANY_VALUE(m.measurement_id) AS measurement_id,
        m.person_id,
        CAST(m.measurement_date AS DATE) AS measurement_date,
        m.unit_concept_id,
        AVG(m.value_as_number) AS value_as_number
    FROM measurement m
    JOIN unit_concept_filter ucf ON m.unit_concept_id = ucf.concept_id
    WHERE m.value_as_number IS NOT NULL --AND m.value_as_number BETWEEN {lower} AND {upper}
    GROUP BY m.person_id, CAST(m.measurement_date AS DATE), m.unit_concept_id
)
SELECT
    d.measurement_id,
    d.value_as_number,
    CASE
        WHEN sex_concept.concept_id = 8507 THEN 0
        WHEN sex_concept.concept_id = 8532 THEN 1
        ELSE 2
    END AS sex,
    EXTRACT(YEAR FROM d.measurement_date) - p.year_of_birth AS age,
    d.unit_concept_id,
    COALESCE(b.bmi, 21.7) AS bmi
FROM deduped d
JOIN person p ON d.person_id = p.person_id
LEFT JOIN gender_concept_filter sex_concept ON p.gender_concept_id = sex_concept.concept_id
LEFT JOIN latest_bmi b ON d.person_id = b.person_id
    """
    
     # Run the query
    values = neuroblu.get_query(query)
    df = pl.DataFrame(values)

    # Filter out-of-range values
    out_of_range = df.filter((pl.col("value_as_number") < lower) | (pl.col("value_as_number") > upper))
    in_range = df.filter((pl.col("value_as_number") >= lower) & (pl.col("value_as_number") <= upper))


    # Save out-of-range values to CSV
    #out_of_csv = f"{measurement}_out_of_range.csv"
    #out_of_range = out_of_range.select(["measurement_id"])
    #out_of_range.write_csv(out_of_csv)
    #print(f"Saved {out_of_range.height} out-of-range rows to {out_of_csv}")

    outlier_df_name = f"{measurement}_outlier_data_frame"
    pd_outlier = out_of_range.to_pandas()
    
    neuroblu.save_df(pd_outlier, outlier_df_name, overwrite=True)
    print(f"Saved {out_of_range.height} outliers to data frame")

    
    # (keep only value_as_number, age, sex)
    in_range = in_range.drop_nulls(subset=["value_as_number", "age", "sex"])

    # Select features
    if measurement == 'body_mass_index':
        X = in_range.select(["value_as_number", "age", "sex"])
    else:
        X = in_range.select(["value_as_number", "age", "sex", "bmi"])

    # Standardize (flatten) to avoid one feature dominating
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.to_numpy())

    # Run model
    df_results, anomalies, scores = isolation_forest(in_range, X_scaled)

    # Save anomalies to CSV
    #anom_csv = f"{measurement}_anomalies.csv"
    #anomalies_columns = anomalies.select(["measurement_id"])
    #anomalies_columns.write_csv(anom_csv)

    anom_df_name = f"{measurement}_anom_date_frame"
    pd_anom = anomalies.to_pandas()

    neuroblu.save_df(pd_anom, anom_df_name, overwrite=True)
    print(f"Saved {anomalies.height} anomalies to data frame")
    

    # Graphs
    graph_distribution(measurement, df_results, anomalies)
    graph_anomalies(measurement, scores)

    # save percentages
    percent = f"{measurement}_make_up.csv"
    message = f"Detected {len(anomalies)} anomalies out of {len(df)} total samples."
    with open(percent, 'w', encoding='utf-8') as f:
        f.write(message)



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
    plt.title(f"{measurement} Isolation Forest Decision Score Distribution")
    plt.xlabel("Decision Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{measurement}_Decision_Score_Distribution.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
