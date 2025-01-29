

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import scipy
import os


###Starting with the data:
butterfly_data = os.getcwd() + "\\Butterfly_Data_adjusted.csv"
plots = pd.read_csv(butterfly_data)
print(plots.columns)

numerical_columns = ['Population denisty (sq km)', 'Average Elevation (m)', 
                    'Average annual precipitiation (mm)', 'Mean temp in C',
                    'Extent of Forest (1000 ha)', 'Number of species', 
                    'area (sq km)', 'latitude']

X = df[numerical_columns]

for col in numerical_columns:
  if X[col].dtype == object:  
      X[col] = X[col].str.replace(',', '', regex=True).astype(float)
  else:
      X[col] = X[col].astype(float) 
    
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  
  return X_scaled, df, numerical_columns
