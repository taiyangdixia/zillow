import numpy as np

a = np.array([1.00019,2])
print a.shape
b = np.vstack((a, a))
c = np.vstack((b, a))
d = np.vstack((c, c))
a = np.around(a, decimals=4)
print a
