import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


columns = [
    "age", "sex", "bmi", "bp",
    "s1", "s2", "s3", "s4", "s5", "s6",
    "target"
]
df = pd.read_csv("C:/Users/USER/OneDrive/Documents/C Programming/ml/potti/diabetes_data.csv",sep=r"\s+",names=columns)
print(df.head())
print(df.columns)
print(df.shape)
print(df.isnull().sum())

