#Drug to Drug Interaction code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "C:/Users/ahana/OneDrive/Desktop/capstone/Drug_to_Drug_Interaction/DDICorpus2013.xlsx"
df = pd.read_excel(file_path)
print(df.head(10))