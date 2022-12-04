import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport


# random seed to reproduce results
random_seed = 458
#np.random.seed(seed=random_seed)

val_size = 0.1
test_size = 0.1
#train_size = 1 - val_size - test_size

# get the data from file
directory = os.getcwd()
data_path = directory + r'/perovskite_data.xlsx'
df = pd.read_excel(data_path, sheet_name="DFT Calculated Dataset")



#%%
# rename some columns to have simpler names
rename_dict = {"Material Composition": "formula",
               "A site #1": "A1",
               "A site #2": "A2",
               "A site #3": "A3",
               "B site #1": "B1",
               "B site #2": "B2",
               "B site #3": "B3",
               "energy_above_hull (meV/atom)": "target",
               "formation_energy (eV/atom)": "Eform"}

df = df.rename(columns=rename_dict)

#%%
""" Splitting data how they did in the paper """

first = ["Ba", "Ca"]

second = ["Pr", "Dy", "Gd", "Ho"]

third = ["Ba", "Sr"]

fourth = ["V", "Cr", "Ti", "Ga", "Sc"]

fifth = ["Bi", "Cd", "Mg", "Ce", "Er"]

#%% set 1: Ba or Ca on the A-site (also mixed both Ba-Ca)
df_test1 = df[df["A1"].isin(first)]
df_test1 = df_test1[df_test1["A2"].isin(first)]
df_val1 = df[df["formula"].isin(df_test1["formula"])==False].sample(round(df.shape[0]*0.1), random_state=42) # take 10% of formulas randomly as validation data
df_train1 = df[(df["formula"].isin(df_test1["formula"])==False)]
df_train1 = df_train1[(df_train1["formula"].isin(df_val1["formula"])==False)]  

test1 = df_test1[["formula", "target"]].copy()
val1 = df_val1[["formula", "target"]].copy()
train1 = df_train1[["formula", "target"]].copy()

#%% set 2: only Pr, dy, Gd, Ho in the A site
df_test2 = df[df["A1"].isin(second)]
df_test2 = df_test2[df_test2["A2"].isin(second)]
df_val2 = df[df["formula"].isin(df_test2["formula"])==False].sample(round(df.shape[0]*0.1), random_state=42)
df_train2 = df[df["formula"].isin(df_test2["formula"])==False & (df["formula"].isin(df_val2["formula"])==False)]
df_train2 = df_train2[(df_train2["formula"].isin(df_val2["formula"])==False)]  

test2 = df_test2[["formula", "target"]].copy()
val2 = df_val2[["formula", "target"]].copy()
train2 = df_train2[["formula", "target"]].copy()

#%% set 3: only Ba AND Sr in A, and Fe in B (needs to have both acording to the size of the train data sets in the data paper)
df_test3Fe = df[(df["B1"]=="Fe") | (df["B2"]=="Fe") | (df["B3"]=="Fe")] # all with Fe somewhere in B (1683 in excel has Fe in B1 and B2)
df_test3 = df_test3Fe[ df_test3Fe["A1"].isin(third) & df_test3Fe["A2"].isin(third) & pd.isnull(df_test3Fe["A3"]) ] # all with Ba and Sr in A1 and A2
df_val3 = df[df["formula"].isin(df_test3["formula"])==False].sample(round(df.shape[0]*0.1), random_state=42)
df_train3 = df[df["formula"].isin(df_test3["formula"])==False & (df["formula"].isin(df_val3["formula"])==False)]
df_train3 = df_train3[(df_train3["formula"].isin(df_val3["formula"])==False)] 

test3 = df_test3[["formula", "target"]].copy()
val3 = df_val3[["formula", "target"]].copy()
train3 = df_train3[["formula", "target"]].copy()
#df_test3 = df_test3Fe[ df_test3Fe["A1"].isin(third) & pd.isnull(df_test3Fe["A2"]) ]    # I would say this is how they describe it in the paper, but that makes too many test elements

#%% set 4: only V, Cr, Ti, Ga, or Sc atoms in B1 and B2
df_test4 = df[df["B1"].isin(fourth)]
df_test4 = df_test4[df_test4["B2"].isin(fourth)]
df_val4 = df[df["formula"].isin(df_test4["formula"])==False].sample(round(df.shape[0]*0.1), random_state=42)
df_train4 = df[df["formula"].isin(df_test4["formula"])==False & (df["formula"].isin(df_val4["formula"])==False)]
df_train4 = df_train4[(df_train4["formula"].isin(df_val4["formula"])==False)] 

test4 = df_test4[["formula", "target"]].copy()
val4 = df_val4[["formula", "target"]].copy()
train4 = df_train4[["formula", "target"]].copy()


#%% set 5: one of the elements in the fifth list somewhere in the A spot
df_test5 = df[ (df["A1"].isin(fifth) & ~(df["A2"].isin(fifth)) & ~(df["A3"].isin(fifth))) |
               (~(df["A1"].isin(fifth)) & df["A2"].isin(fifth) & ~(df["A3"].isin(fifth))) |
               (~(df["A1"].isin(fifth)) & ~(df["A2"].isin(fifth)) & df["A3"].isin(fifth))]
               
df_val5 = df[df["formula"].isin(df_test5["formula"])==False].sample(round(df.shape[0]*0.1), random_state=42)
df_train5 = df[df["formula"].isin(df_test5["formula"])==False & (df["formula"].isin(df_val5["formula"])==False)]
df_train5 = df_train5[(df_train5["formula"].isin(df_val5["formula"])==False)] 

test5 = df_test5[["formula", "target"]].copy()
val5 = df_val5[["formula", "target"]].copy()
train5 = df_train5[["formula", "target"]].copy()
#%%
""" Splitting the data randomly """

df_test6 = df.sample(frac=test_size, random_state=(random_seed))
df_val6 = df.drop(df_test6.index).sample(frac=val_size, random_state=(random_seed))
df_train6 = df.drop(df_test6.index).drop(df_val6.index)


test6 = df_test6[["formula", "target"]].copy()
val6 = df_val6[["formula", "target"]].copy()
train6 = df_train6[["formula", "target"]].copy()
#%% profiling of the data

# data is already checked manually for weird things with the profile
#profile = ProfileReport(df, title="data profile")
#profile.to_file("report.html")
"""
profileTest1 = ProfileReport(df_test1, title="test 1")
profileTrain1 = ProfileReport(df_train1, title="train 1")
profileTest1.to_file("test1.html")
profileTrain1.to_file("train1.html")

profileTest2 = ProfileReport(df_test2, title="test 2")
profileTrain2 = ProfileReport(df_train2, title="train 2")
profileTest2.to_file("test2.html")
profileTrain2.to_file("train2.html")

profileTest3 = ProfileReport(df_test3, title="test 3")
profileTrain3 = ProfileReport(df_train3, title="train 3")
profileTest3.to_file("test3.html")
profileTrain3.to_file("train3.html")

profileTest4 = ProfileReport(df_test4, title="test 4")
profileTrain4 = ProfileReport(df_train4, title="train 4")
profileTest4.to_file("test4.html")
profileTrain4.to_file("train4.html")

profileTest5 = ProfileReport(df_test5, title="test 5")
profileTrain5 = ProfileReport(df_train5, title="train 5")
profileTest5.to_file("test5.html")
profileTrain5.to_file("train5.html")

profileTest6 = ProfileReport(df_test6, title="test 6")
profileTrain6 = ProfileReport(df_train6, title="train 6")
profileTest6.to_file("test6.html")
profileTrain6.to_file("train6.html")


# get unique element in first spot of molecule
#unique_molecule = df["A1"].unique()
#print(f'{len(unique_molecule)} unique formulae:\n{unique_molecule}')
"""

#%%
""" Saving data """
# uncomment to save data again

train_path1 = directory + '/train_data1.csv'
test_path1 = directory + '/test_data1.csv'
val_path1 = directory + '/val_data1.csv'
train1.to_csv(train_path1, index=False)
test1.to_csv(test_path1, index=False)
val1.to_csv(val_path1, index=False)

train_path2 = directory + '/train_data2.csv'
test_path2 = directory + '/test_data2.csv'
val_path2 = directory + '/val_data2.csv'
test2.to_csv(test_path2, index=False)
train2.to_csv(train_path2, index=False)
val2.to_csv(val_path2, index=False)

train_path3 = directory + '/train_data3.csv'
test_path3 = directory + '/test_data3.csv'
val_path3 = directory + '/val_data3.csv'
test3.to_csv(test_path3, index=False)
train3.to_csv(train_path3, index=False)
val3.to_csv(val_path3, index=False)

train_path4 = directory + '/train_data4.csv'
test_path4 = directory + '/test_data4.csv'
val_path4 = directory + '/val_data4.csv'
test4.to_csv(test_path4, index=False)
train4.to_csv(train_path4, index=False)
val4.to_csv(val_path4, index=False)

train_path5 = directory + '/train_data5.csv'
test_path5 = directory + '/test_data5.csv'
val_path5 = directory + '/val_data5.csv'
test5.to_csv(test_path5, index=False)
train5.to_csv(train_path5, index=False)
val5.to_csv(val_path5, index=False)

train_path6 = directory + '/train_data6.csv'
test_path6 = directory + '/test_data6.csv'
val_path6 = directory + '/val_data6.csv'
test6.to_csv(test_path6, index=False)
train6.to_csv(train_path6, index=False)
val6.to_csv(val_path6, index=False)


print("done")
