import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport


# random seed to reproduce results
random_seed = 458
#np.random.seed(seed=random_seed)

#val_size = 0.2
t_size = 0.2
#train_size = 1 - val_size - test_size

# get the data from file
directory = os.getcwd()
data_path = directory + '/perovskite_data.xlsx'
df = pd.read_excel(data_path, sheet_name="DFT Calculated Dataset")


# rename some columns to have simpler names
rename_dict = {"Material Composition": "formula",
               "A site #1": "A1",
               "A site #2": "A2",
               "A site #3": "A3",
               "B site #1": "B1",
               "B site #2": "B2",
               "B site #3": "B3",
               "energy_above_hull (meV/atom)": "Ehull",
               "formation_energy (eV/atom)": "Eform"}

df = df.rename(columns=rename_dict)

""" Splitting data how they did in the paper """

first = ["Ba", "Ca"]

second = ["Pr", "Dy", "Gd", "Ho"]

thirdA = ["Ba", "Sr"]
thirdB = ["Fe"]

fourth = ["V", "Cr", "Ti", "Ga", "Sc"]

fifth = ["Bi", "Cd", "Mg", "Ce", "Er"]


df_test1 = df[df["A1"].isin(first)]
df_test1 = df_test1[df_test1["A2"].isin(first)]

df_test2 = df[df["A1"].isin(second)]
df_test2 = df_test2[df_test2["A2"].isin(second)]

df_test3A = df[df["B1"].isin(thirdB)]
df_test3B = df[df["B2"].isin(thirdB)]
df_test3C = df[df["B3"].isin(thirdB)]
df_test3 = pd.concat([df_test3A, df_test3B, df_test3C])
df_test3 = df_test3[df_test3["A1"].isin(thirdA)]
df_test3 = df_test3[df_test3["A2"].isin(thirdA)]

df_test4 = df[df["B1"].isin(fourth)]
df_test4 = df_test4[df_test4["B2"].isin(fourth)]

df_test5 = df[df["A1"].isin(fifth)]
df_test5 = df[df["A2"].isin(fifth)]
df_test5 = df[df["A3"].isin(fifth)]


df_train5 = df[df["formula"].isin(df_test5["formula"])==False]

""" Splitting the data randomly """
# splitting with sklearn randomly might work since there are only unique formulas
#X = df[['formula']]
#y = df['Ehull']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size, random_state=random_seed)

#print("input\n")
#print(X_train.head(10))
#print("\noutput\n")
#print(y_train.head(10))

# data is already checkd manually for weird things with the profile
#profile = ProfileReport(df, title="data profile")
#profile.to_file("report.html")

# get unique element in first spot of molecule
#unique_molecule = df["A1"].unique()
#print(f'{len(unique_molecule)} unique formulae:\n{unique_molecule}')


""" Splitting the data with one element as test """
#val_element = "Ba"
#test_element = "La"

# split the data so that the train data doens't contain any of the test element

#df_val = df[df["formula"].str.contains(val_element)]
#df_test = df[df["formula"].str.contains(test_element)]
#df_train = df[df["formula"].str.contains(test_element)==False]


#print(f'train dataset shape: {df_train.shape}')
#print(f'validation dataset shape: {df_val.shape}')
#print(f'test dataset shape: {df_test.shape}\n')


""" Saving data """
# uncomment to save data again
#train_path = directory + 'train_data.csv'
#test_path = directory + 'test_data.csv'
#df_train.to_csv(train_path, index=False)
#df_test.to_csv(test_path, index=False)


# make reports for test and train data
#profile_test = ProfileReport(df_test, title="data profile")
#profile_test.to_file("test_report.html")

#profile_train = ProfileReport(df_train, title="data profile")
#profile_train.to_file("train_report.html")

