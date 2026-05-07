# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# %%
df= pd.read_csv("Titanic-Dataset.csv")
df.head()

# %%
df.describe()

# %%
df.info()

# %%
df['Age'] = df['Age'].astype(float)

# %%
df.shape

# %%
df.isnull().sum()

# %%
df['Sex'] = df['Sex'].astype('string')

# %%
df.info()

# %%
df.dtypes

# %%
scaler = MinMaxScaler()
df['Age_norm'] = scaler.fit_transform(df[['Age']])

# %%
df.info()

# %%
df['Age_norm2'] = (df['Age'] - df['Age'].min()) / (df['Age'].max() - df['Age'].min())

# %%
df[['Age_norm2', 'Age_norm']]

# %%
df2 = pd.get_dummies(df, columns = ['Embarked'])

# %%
df2.head()

# %%
le = LabelEncoder()
df['Embarked_enc'] = le.fit_transform(df['Embarked'])

# %%
df.head()

# %%




