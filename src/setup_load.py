# Step 1: Setup - Import libraries, load data, initial look

import pandas as pd
import numpy as np

# Load dataset
FILE_PATH = "./data/customer_churn_data.csv"   # adjust if needed

df = pd.read_csv(FILE_PATH)

print("\n========== Dataset Loaded Successfully ==========\n")
print("Rows & Columns:", df.shape)

print("\n========== Column Info ==========\n")
print(df.info())

print("\n========== Summary Statistics ==========\n")
print(df.describe(include='all'))

print("\n========== Sample Rows ==========\n")
print(df.head(10))   # using print instead of display()
