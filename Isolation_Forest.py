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
    
    # ['measurement', 'concept_id', 'lower', 'upper', 'contamination']
    measurements = [
        ['temperature', (4302666,), 95, 100.4], #0
        ['body_mass_index', (4245997,), 17, 29.9], #1
        ['diastolic_blood_pressure', (4154790,), 50, 89, 0.05], #2
        ['systolic_blood_pressure', (4152194,), 80, 139, 0.05], #3
        ['pulse_rate', (4301868,), 40, 120], #4
        ['pulse_oximetry', (4098046,), 85, 100], #5
        ['fasting_glucose', (3037110,), 50, 125, 0.0975], #6
        ['A1c', (3003309, 3004410, 3005673, 3007263, 3034639, 36032094, 40762352, 42869630), 3.5, 6.4, 0.07], #7
        #['body_weight', (4099154,), 0, 1000],
        #['body_height', (4177340,), 0, 275],
    ]
    
     # Run the query
    measurement, concept_id, lower, upper = measurements[selector]
    query = query_function(concept_id, lower, upper)
    values = neuroblu.get_query(query)
    df = pl.DataFrame(values)
    
    # (keep only value_as_number, age, sex, BMI)
    if measurement == 'body_mass_index':
        X = df.select(["value_as_number", "age", "sex"])
    else:
        X = df.select(["value_as_number", "age", "sex", "bmi"])

    # Standardize (flatten) to avoid one feature dominating
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.to_numpy())

    # Run model
    df_results, anomalies, scores = isolation_forest(X, X_scaled)

    #save anomalies to data frame
    anom_df_name = f"{measurement}_anom_data_frame"
    pd_anom = anomalies.to_pandas()
    neuroblu.save_df(pd_anom, anom_df_name, overwrite=True)
    print(f"Saved {anomalies.height} anomalies to data frame")
    

    # Graphs
    P = anomalies.height / df.height
    percentage = f"Detected {anomalies.height} anomalies out of {df.height} total samples ({P:.2%})."
    if selector == 1:
        graph_distribution_bmi_only(df_results, anomalies, percentage)
    else:
        graph_distribution(measurement, df, anomalies, percentage)



def isolation_forest(df, X_scaled, contamination):
    iso = IsolationForest(contamination='auto', random_state=42)
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


def graph_distribution(measurement, df, anomalies, percentage):
    # Age Graph
    plt.figure(figsize=(10, 6))
    plt.hexbin(anomalies["value_as_number"], anomalies["age"], gridsize=50, cmap="Reds", mincnt=1)
    plt.colorbar(label="Count")
    plt.title(f"{measurement}: Value vs Age\n{percentage}")
    plt.xlabel("value_as_number")
    plt.ylabel("Age")
    plt.tight_layout()
    plt.savefig(f"{measurement}_value_vs_age.png", dpi=300)
    plt.close()

    # Sex Graph (jitter to avoid overlap)
    plt.figure(figsize=(10, 6))
    sex_jitter = anomalies["sex"] + np.random.uniform(-0.1, 0.1, size=len(anomalies))
    plt.scatter(anomalies["value_as_number"], sex_jitter, s=8, alpha=0.5)
    plt.yticks([0, 1, 2], ["Male", "Female", "Other"])
    plt.title(f"{measurement}: Value vs Sex\n{percentage}")
    plt.xlabel("value_as_number")
    plt.ylabel("Sex")
    plt.tight_layout()
    plt.savefig(f"{measurement}_value_vs_sex.png", dpi=300)
    plt.close()

    # BMI graph (hexbin)
    plt.figure(figsize=(10, 6))
    plt.hexbin(anomalies["value_as_number"], anomalies["bmi"], gridsize=50, cmap="Reds", mincnt=1)
    plt.colorbar(label="Count")
    plt.title(f"{measurement}: Value vs BMI\n{percentage}")
    plt.xlabel("value_as_number")
    plt.ylabel("BMI")
    plt.tight_layout()
    plt.savefig(f"{measurement}_value_vs_bmi.png", dpi=300)
    plt.close()

    # 4D graph (sex markers + BMI color)
    plt.figure(figsize=(12, 7))
    markers = {0: "o", 1: "s", 2: "D"}
    bmi_min = float(anomalies["bmi"].min())
    bmi_max = float(anomalies["bmi"].max())
    norm = plt.Normalize(vmin=bmi_min, vmax=bmi_max)

    for sex_value, marker in markers.items():
        subset = anomalies.filter(pl.col("sex") == sex_value).to_pandas()
        if len(subset) == 0:
            continue
        plt.scatter(
            subset["value_as_number"],
            subset["age"],
            c=subset["bmi"],
            cmap="coolwarm",
            norm=norm,
            s=30,
            alpha=0.6,
            marker=marker,
            label=f"Sex {sex_value}"
        )

    cbar = plt.colorbar()
    cbar.set_label("BMI")
    plt.xlabel("value_as_number")
    plt.ylabel("Age")
    plt.title(f"{measurement} — Multi-Dimensional Plot\n{percentage}")
    plt.legend(title="Sex")
    plt.tight_layout()
    plt.savefig(f"{measurement}_4d_plot.png", dpi=300)
    plt.close()


def graph_distribution_bmi_only(df, anomalies, percentage):
    # Age vs BMI (hexbin)
    plt.figure(figsize=(10, 6))
    plt.hexbin(anomalies["value_as_number"], anomalies["age"], gridsize=50, cmap="Reds", mincnt=1)
    plt.colorbar(label="Count")
    plt.title(f"BMI: BMI vs Age\n{percentage}")
    plt.xlabel("BMI")
    plt.ylabel("Age")
    plt.tight_layout()
    plt.savefig("bmi_vs_age.png", dpi=300)
    plt.close()

    # Sex vs BMI (jitter)
    plt.figure(figsize=(10, 6))
    sex_jitter = anomalies["sex"] + np.random.uniform(-0.1, 0.1, size=len(anomalies))
    plt.scatter(anomalies["value_as_number"], sex_jitter, s=8, alpha=0.5)
    plt.yticks([0, 1, 2], ["Male", "Female", "Other"])
    plt.title(f"BMI: BMI vs Sex\n{percentage}")
    plt.xlabel("BMI")
    plt.ylabel("Sex")
    plt.tight_layout()
    plt.savefig("bmi_vs_sex.png", dpi=300)
    plt.close()

    # BMI vs Age with sex-coded markers
    plt.figure(figsize=(12, 7))
    markers = {0: "o", 1: "s", 2: "D"}
    for sex_value, marker in markers.items():
        subset = anomalies.filter(pl.col("sex") == sex_value).to_pandas()
        if len(subset) == 0:
            continue
        plt.scatter(
            subset["value_as_number"],
            subset["age"],
            marker=marker,
            alpha=0.6,
            s=30,
            label=f"Sex {sex_value}",
        )
    plt.xlabel("BMI")
    plt.ylabel("Age")
    plt.title(f"BMI — BMI vs Age (Sex-coded markers)\n{percentage}")
    plt.legend(title="Sex")
    plt.tight_layout()
    plt.savefig("bmi_bmi_vs_age_sex_markers.png", dpi=300)
    plt.close()


def query_function(concept_id, lower, upper):

    query = f"""
    WITH
    unit_concept_filter AS (
        SELECT DISTINCT c.concept_id, c.concept_name
        FROM concept c
        JOIN measurement m
            ON c.concept_id = m.unit_concept_id
        WHERE m.measurement_concept_id IN {concept_id}
    ),
    gender_concept_filter AS (
        SELECT concept_id, concept_name
        FROM concept
        WHERE concept_id IN (8507, 8532)
    ),
    average_measurements_per_date AS (
        SELECT 
            m.person_id,
            m.measurement_date,
            AVG(m.value_as_number) AS value_as_number,
            m.unit_concept_id
        FROM measurement m
        WHERE m.measurement_concept_id IN {concept_id}
          AND m.value_as_number BETWEEN {lower} AND {upper}
          AND m.value_as_number IS NOT NULL
          AND m.unit_concept_id IN (SELECT concept_id FROM unit_concept_filter)
        GROUP BY m.person_id, m.measurement_date, m.unit_concept_id
    ),
    patient_last_date AS (
        SELECT 
            person_id,
            MAX(measurement_date) AS max_measurement_date
        FROM average_measurements_per_date
        GROUP BY person_id
    ),
    measurement_extract AS (
        SELECT a.*
        FROM average_measurements_per_date a
        JOIN patient_last_date b
            ON a.person_id = b.person_id
           AND a.measurement_date = b.max_measurement_date
    ),
    bmi_values AS (
        SELECT
            person_id,
            measurement_date,
            value_as_number AS bmi
        FROM measurement
        WHERE measurement_concept_id = 4245997
          AND value_as_number IS NOT NULL
    ),
    latest_bmi AS (
        SELECT DISTINCT ON (person_id)
            person_id,
            bmi
        FROM bmi_values
        ORDER BY person_id, measurement_date DESC
    )
    
    SELECT
        m.person_id,
        m.value_as_number,
        CASE
            WHEN sex_concept.concept_id = 8507 THEN 0
            WHEN sex_concept.concept_id = 8532 THEN 1
            ELSE 2
        END AS sex,
        EXTRACT(YEAR FROM m.measurement_date) - p.year_of_birth AS age,
        m.unit_concept_id,
        COALESCE(latest_bmi.bmi, 21.7) AS bmi
    FROM measurement_extract m
    JOIN person p 
        ON m.person_id = p.person_id
    LEFT JOIN gender_concept_filter sex_concept 
        ON p.gender_concept_id = sex_concept.concept_id
    LEFT JOIN latest_bmi 
        ON m.person_id = latest_bmi.person_id
    """
    return query

if __name__ == "__main__":
    main()
