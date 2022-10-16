from cgi import test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pandas_profiling import ProfileReport

# random seed to reproduce results
#random_seed = 458
#np.random.seed(seed=random_seed)

#val_size = 0.2
#test_size = 0.1
#train_size = 1 - val_size - test_size

# get the data from file
PATH = os.getcwd()
data_path = os.path.join(PATH, '/Users/carlc/OneDrive/Desktop/B.S.-Thesis/perovskite_data.xlsx')
df = pd.read_excel(data_path, sheet_name="DFT Calculated Dataset")


# rename some columns to have simpler names
rename_dict = {"Material Composition": "formula",
               "A site #1": "A1",
               "A site #2": "A2",
               "A site #3": "A3",
               "B site #1": "B1",
               "energy_above_hull (meV/atom)": "Ehull",
               "formation_energy (eV/atom)": "Eform"}

df = df.rename(columns=rename_dict)

# data is already checkd manually for weird things with the profile
#profile = ProfileReport(df, title="data profile")
#profile.to_file("report.html")

# get unique element in first spot of molecule
#unique_molecule = df["A1"].unique()
#print(f'{len(unique_molecule)} unique formulae:\n{unique_molecule}')

#val_element = "Ba"
test_element = "La"

# split the data so that the train data doens't contain any of the test element

#df_val = df[df["formula"].str.contains(val_element)]
df_test = df[df["formula"].str.contains(test_element)]
df_train = df[df["formula"].str.contains(test_element)==False]


#print(f'train dataset shape: {df_train.shape}')
#print(f'validation dataset shape: {df_val.shape}')
#print(f'test dataset shape: {df_test.shape}\n')


# uncomment to save data again
#train_path = os.path.join(PATH, '/Users/carlc/OneDrive/Desktop/B.S.-Thesis/train_data.csv')
#test_path = os.path.join(PATH, '/Users/carlc/OneDrive/Desktop/B.S.-Thesis/test_data.csv')
#df_train.to_csv(train_path, index=False)
#df_test.to_csv(test_path, index=False)


# make reports for test and train data
#profile_test = ProfileReport(df_test, title="data profile")
#profile_test.to_file("test_report.html")

#profile_train = ProfileReport(df_train, title="data profile")
#profile_train.to_file("train_report.html")

