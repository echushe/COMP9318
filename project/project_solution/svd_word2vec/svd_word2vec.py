# Singular Value Decomposition Example

import numpy as np

# example taken from Video Tutorials - All in One
# https://www.youtube.com/watch?v=P5mlg91as1c
a = np.array([[1, 1, 1, 0, 0],
              [3, 3, 3, 0, 0],
              [4, 4, 4, 0, 0],
              [5, 5, 5, 0, 0],
              [0, 2, 0, 4, 4],
              [0, 0, 0, 5, 5],
              [0, 1, 0, 2, 2]])

# set numpy printing options
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

# Full SVD is taught more often. Here is a good explination of the different
# http://www.cs.cornell.edu/Courses/cs322/2008sp/stuff/TrefethenBau_Lec4_SVD.pdf
print ("--- FULL ---")
U, s, VT = np.linalg.svd(a, full_matrices=True)

print ("U:\n {}".format(U))
print ("s:\n {}".format(s))
print ("VT:\n {}".format(VT))

# the reduced or trucated SVD operation can save time by ignoring all the
# extremly small or exactly zero values. A good blog post explaing the benefits
# can be found here:
# http://blog.explainmydata.com/2016/01/how-much-faster-is-truncated-svd.html
print ("--- REDUCED ---")

U, s, VT = np.linalg.svd(a, full_matrices=False)

print ("U:\n {}".format(U))
print ("s:\n {}".format(s))
print ("VT:\n {}".format(VT))




