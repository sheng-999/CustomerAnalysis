import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# import data
df = pd.read_csv('Dataset test technique FRINGUANT - Data Analyst.csv')

# Discovery of Data
print(df.head())
print(df.info())
print(df.describe())
