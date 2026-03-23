# import packages
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata
from scipy.stats import wasserstein_distance, ks_2samp, chi2_contingency, fisher_exact
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import kaleido
# set and prep data

DATA_PATH = r"C:/Users/c3058452/OneDrive - Newcastle University/Work in Progress/Saved_Rdata/testdata3.csv"
OUT_DIR = r"C:/Users/c3058452/OneDrive - Newcastle University/Work in Progress/CTGAN"
os.makedirs(OUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_PATH)

# select relevant columns
cols = ['Sex', 'Outcome', 'Age', 'Ethnicity', 'Location', 'Years_treatment', 'Treat']
df = df[cols].copy()

# drop NAs (full row)
df = df.dropna(subset=cols).reset_index(drop=True)

# label data types
continuous_features = ['Age', 'Years_treatment']
binary_features = ['Sex', 'Outcome']
categorical_features = ['Ethnicity', 'Location']
condition_col = 'Treat'


# create metadata description
metadata = Metadata.detect_from_dataframe(data=df)

# save for future use
metadata.save_to_json('test_metadata3.json') #reset every run

ctgan = CTGANSynthesizer(
    metadata=metadata,
    enforce_rounding=True, # synth data has same number of decimal digits as real = TRUE
    enforce_min_max_values=True, # synth data has same min/max boundaries as real = TRUE
    epochs = 200,
    verbose=True
)

print("Fitting CTGAN... this may take some minutes")
ctgan.fit(df)

# save model
model_path = os.path.join(OUT_DIR, "ctgan_model.pkl")
ctgan.save(model_path)
print("Saved CTGAN model to:", model_path)

ctgan.get_loss_values()
fig = ctgan.get_loss_values_plot()
fig.show()



