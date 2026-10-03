import pandas as pd

df = pd.read_csv("data/filtered_labels.csv")
print(df["target"].value_counts())