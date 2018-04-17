import csv
import string
import random
import pandas as pd
import numpy as np

cols = 9
rows = 256
col_names = []
for i in range(cols):
    col_names.append(''.join(random.choices(string.ascii_uppercase + string.digits, k=2)))

random_df = pd.DataFrame(columns = col_names)
row = np.zeros((cols), dtype = object)

for i in range(rows):

    col_val = i
    for j in range(cols - 1):
        row[j] = (col_val % 2) * 999999
        col_val //= 2

    row[-1] = 1 # random.randint(0, 300)

    random_df.loc[i] = row

random_df.to_csv('random_csv.txt', sep = '\t', index=False)
