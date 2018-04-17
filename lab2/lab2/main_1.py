import numpy as np
import submission as submission
import copy

'''
array = np.array([1, 2, 3, 4, 5, 6], dtype=object)
original = copy.deepcopy(array)

print(array)
while submission.list_plus(array, original):
    print(array)
'''
input_data = submission.read_data('random_csv.txt')
'''
ret = submission.single_tuple_optimization(input_data.iloc[:1,:])
print(ret)
'''

output = submission.buc_rec_optimized(input_data)
print(output)
output.to_csv('output.txt', sep = '\t', index=False)

