import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# 1. Read in the two CSV files
df11 = pd.read_csv("implizite_Analyse/results/run_1_1/scoring_model_model.csv")
df12 = pd.read_csv("implizite_Analyse/results/run_1_2/scoring_model_model.csv")
df13 = pd.read_csv("implizite_Analyse/results/run_1_3/scoring_model_model.csv")
df21 = pd.read_csv("implizite_Analyse/results/run_2_1/scoring_model_model.csv")
df22 = pd.read_csv("implizite_Analyse/results/run_2_2/scoring_model_model.csv")
df23 = pd.read_csv("implizite_Analyse/results/run_2_3/scoring_model_model.csv")

# Concatenate the two dataframes
#df = pd.concat([df1, df2, df3], ignore_index=True)
df = pd.concat([df11, df12, df13,df21,df22,df23], ignore_index=True)
print("Columns in df11:", df11.columns.tolist())
print("Columns in df12:", df12.columns.tolist())
print("Columns in df13:", df13.columns.tolist())
print("Columns in df21:", df21.columns.tolist())
print("Columns in df22:", df22.columns.tolist())
print("Columns in df23:", df23.columns.tolist())
print("Columns in concatenated df:", df.columns.tolist())


# 2. Create a merged dataframe by grouping by Model, Group, and Axis Name,
#    and computing the mean for 'mean' and 'SEM'
merged_df = df.groupby(["Scorer Model", "Source Model"], as_index=False).agg({"Count": "sum", "Score": "mean", "SEM": "mean"})

# Save the merged csv file
output_dir = "implizite_Analyse/results/combined"
os.makedirs(output_dir, exist_ok=True)
merged_csv_filename = os.path.join(output_dir, "combined_model_model.csv")
merged_df.to_csv(merged_csv_filename, index=False)
print(f"Saved merged CSV: {merged_csv_filename}")
