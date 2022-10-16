import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pandas_profiling import ProfileReport

PATH = os.getcwd()
data_path = os.path.join(PATH, '/Users/carlc/OneDrive/Desktop/B.S.-Thesis/perovskite_data.xlsx')
df = pd.read_excel(data_path, sheet_name="DFT Calculated Dataset")

#profile = ProfileReport(df, title="data profile")
#profile.to_file("report.html")


