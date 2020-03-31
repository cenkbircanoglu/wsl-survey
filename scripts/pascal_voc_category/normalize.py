import numpy as np
from sklearn.preprocessing import StandardScaler
X = np.array([[4, 1, 2, 2],
      [1, 3, 9, 3],
     [5, 7, 5, 1]])

row_sums = X.sum()
new_matrix = X / row_sums[:, np.newaxis]
print(new_matrix)
