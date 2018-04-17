import numpy as np
import submission as submission
import copy
import random
import time


x = [3, 1, 18, 11, 13, 17]
num_bins = 4
matrix, bins = submission.v_opt_dp(x, num_bins)

for row in matrix:
    print(row)

print(bins)


x = [1, 2, 3, 4, 5, 7, 10]
num_bins = 3
matrix, bins = submission.v_opt_dp(x, num_bins)

for row in matrix:
    print(row)

print(bins)



random.seed(0)
for step in range(1, 11):
    x = []
    num_bins = 20 * step
    for i in range(100 * step):
        x.append(random.randint(0, 100))

    # print('x = {}'.format(x))

    time1 = int(round(time.time() * 1000))

    matrix, bins = submission.v_opt_dp(x, num_bins)

    time2 = int(round(time.time() * 1000))

    '''
    for row in matrix:
        print(row)
    '''

    # print(bins)
    print('List size: {}\tBin number: {}'.format(100 * step, 20 * step))
    print('Time used: {}'.format(time2 - time1))