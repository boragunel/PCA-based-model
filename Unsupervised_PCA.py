

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