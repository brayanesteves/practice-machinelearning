import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target

# Output:
#    sepal length (cm)  sepal width (cm)  ...  petal width (cm)  species
# 0                5.1               3.5  ...               0.2        0
# 1                4.9               3.0  ...               0.2        0
# 2                4.7               3.2  ...               0.2        0
# 3                4.6               3.1  ...               0.2        0
# 4                5.0               3.6  ...               0.2        0
print(df.head())

# Output:
# <class 'pandas.DataFrame'>
# RangeIndex: 150 entries, 0 to 149
# Data columns (total 5 columns):
 #   Column             Non-Null Count  Dtype  
# ---  ------             --------------  -----  
#  0   sepal length (cm)  150 non-null    float64
#  1   sepal width (cm)   150 non-null    float64
#  2   petal length (cm)  150 non-null    float64
#  3   petal width (cm)   150 non-null    float64
#  4   species            150 non-null    int64  
# dtypes: float64(4), int64(1)
# memory usage: 6.0 KB
# None
print(df.info())

# Output:
# sepal length (cm)    0
# sepal width (cm)     0
# petal length (cm)    0
# petal width (cm)     0
# species              0
# dtype: int64
print(df.isnull().sum())

# Output:
#        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)     species
# count         150.000000        150.000000         150.000000        150.000000  150.000000
# mean            5.843333          3.057333           3.758000          1.199333    1.000000
# std             0.828066          0.435866           1.765298          0.762238    0.819232
# min             4.300000          2.000000           1.000000          0.100000    0.000000
# 25%             5.100000          2.800000           1.600000          0.300000    0.000000
# 50%             5.800000          3.000000           4.350000          1.300000    1.000000
# 75%             6.400000          3.300000           5.100000          1.800000    2.000000
# max             7.900000          4.400000           6.900000          2.500000    2.000000
print(df.describe())

encoder = LabelEncoder()
df["species"] = encoder.fit_transform(df["species"])

scaler_minmax = MinMaxScaler()
df_norm = pd.DataFrame(scaler_minmax.fit_transform(df.iloc[:, :-1]), columns=iris.feature_names)

# [ENGLISH]
# Calculates the mean and standard deviation of each column, then transforms each value.
# [SPANISH]
# Calcula la media y desviacion de caca columna, luego transforma cada valor.
scaler_std = StandardScaler()
df_std = pd.DataFrame(scaler_std.fit_transform(df.iloc[:, :-1]), columns=iris.feature_names)

comparison = pd.DataFrame()
for col in iris.feature_names:
    comparison[col + " (Original)"] = df[col]
    comparison[col + " (Normalizado)"] = df_norm[col]
    comparison[col + " (Estandarizado)"] = df_std[col]

comparison["species"] = df["species"]
# Output:
#      sepal length (cm) (Original)  sepal length (cm) (Normalizado)  sepal length (cm) (Estandarizado)  sepal width (cm) (Original)  ...  petal width (cm) (Original)  petal width (cm) (Normalizado)  petal width (cm) (Estandarizado)  species
# 0                             5.1                         0.222222                          -0.900681                          3.5  ...                          0.2                        0.041667                         -1.315444        0
# 1                             4.9                         0.166667                          -1.143017                          3.0  ...                          0.2                        0.041667                         -1.315444        0
# 2                             4.7                         0.111111                          -1.385353                          3.2  ...                          0.2                        0.041667                         -1.315444        0
# 3                             4.6                         0.083333                          -1.506521                          3.1  ...                          0.2                        0.041667                         -1.315444        0
# 4                             5.0                         0.194444                          -1.021849                          3.6  ...                          0.2                        0.041667                         -1.315444        0
# ..                            ...                              ...                                ...                          ...  ...                          ...                             ...                               ...      ...
# 145                           6.7                         0.666667                           1.038005                          3.0  ...                          2.3                        0.916667                          1.448832        2
# 146                           6.3                         0.555556                           0.553333                          2.5  ...                          1.9                        0.750000                          0.922303        2
# 147                           6.5                         0.611111                           0.795669                          3.0  ...                          2.0                        0.791667                          1.053935        2
# 148                           6.2                         0.527778                           0.432165                          3.4  ...                          2.3                        0.916667                          1.448832        2
# 149                           5.9                         0.444444                           0.068662                          3.0  ...                          1.8                        0.708333                          0.790671        2
# 
# [150 rows x 13 columns]
print(comparison)