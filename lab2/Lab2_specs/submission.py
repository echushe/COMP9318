## import modules here 
import pandas as pd
import numpy as np


################# Question 1 #################
def read_data(filename):
    df = pd.read_csv(filename, sep='\t')
    return (df)

# helper functions
def project_data(df, d):
    # Return only the d-th column of INPUT
    return df.iloc[:, d]

def select_data(df, d, val):
    # SELECT * FROM INPUT WHERE input.d = val
    col_name = df.columns[d]
    return df[df[col_name] == val]

def remove_first_dim(df):
    # Remove the first dim of the input
    return df.iloc[:, 1:]

def slice_data_dim0(df, v):
    # syntactic sugar to get R_{ALL} in a less verbose way
    df_temp = select_data(df, 0, v)
    return remove_first_dim(df_temp)


def list_plus(array, original_array):
    """
    In this function we treat the list as a binary digital number
    The elements that are not 'ALL' in the list are recognized as 0
    The elements that are 'ALL' or '*' in the list are considered as 1
    for example:
    [ 1, 5, 7 ] is the original list
    [ 1, 5, '*' ] can be treated as 001, then,
    if we plus one to this list, this list will become [ 1, '*', 7 ]
    that resembles 010
    """
    i = len(array) - 1
    while i >= 0:
        if array[i] != 'ALL':
            array[i] = 'ALL'
            return True
        else:
            array[i] = original_array[i]

        i -= 1

    return False



def single_tuple_optimization(df):

    # Get data value of this single tuple
    value = df.iloc[0,-1]
    # Remove the last dim
    pure_dimenions = df.iloc[0, : -1]
    # convert the tuple to a numpy array
    array = np.zeros(df.values.shape[1] - 1, dtype = object)
    original = np.copy(pure_dimenions.values)

    for i in range(len(array)):
        array[i] = original[i]

    ret = []
    ret.append(df.values[0])

    while list_plus(array, original):       
        
        row = np.zeros(df.values.shape[1], dtype = object)

        for i in range(len(array)):
            row[i] = array[i]

        row[-1] = value
        ret.append(row)

    return ret


def buc_rec_optimized(df):# do not change the heading of the function
    
    df_to_fill = pd.DataFrame(columns = df.columns)
    dict_buffer = {}

    index = 0
    for i in range(df.values.shape[0]):
        rows = single_tuple_optimization(df.iloc[i:i+1, :])
        
        for row in rows:
            value = row[-1]
            key = ""

            for j in range(len(row) - 1):
                key += str(row[j])
                if j < len(row) - 2:
                    key += '~'
            
            if key in dict_buffer:
                dict_buffer[key] += int(value)
            else:
                dict_buffer[key] = int(value)

    index = 0
    for key, value in sorted(dict_buffer.items()):

        str_list = key.split('~')
        new_row = np.zeros((len(str_list) + 1), dtype = object)
        
        for i in range(len(str_list)):
                new_row[i] = str_list[i]

        new_row[-1] = value

        # print(new_row)

        df_to_fill.loc[index] = new_row
        index += 1
            
    return df_to_fill


################# Question 2 #################
class opt_mat:
    def __init__(self):
        self.val = 0
        self.sse = []
        self.left_side_eles = []
        

def sse(x):
    """
    Compute Standard Square Error of a list x
    """
    mean = sum(x) / len(x)
    s = 0.0
    
    for ele in x:
        sub = ele - mean
        s += sub * sub
    return s


def build_sse_cache(x, num_bins):
    """
    Some SSE values should be calculated fromn scratch
    We should cache these SSE values to speed up the algorithm
    """
    x = x[::-1]

    max_ran = len(x) - num_bins + 1
    cache = []
    for i in range(len(x)):
        row = []
        
        for j in range(max_ran):
            start = i - j
            if start < 0:
                break;
            row.append(sse(x[start : i + 1]))
        
        cache.append(row)

    return cache


class opt_matrix_node:
    def __init__(self, sse_range, prev_pos, val):
        self.sse_range = sse_range
        self.prev_pos = prev_pos
        self.val = val


def v_opt_dp(x, num_bins):# do not change the heading of the function

    min_sse_mat = []
    x_len = len(x)
    sse_cache = build_sse_cache(x, num_bins)

    for i in range(num_bins):
        row = []
        for j in range(x_len):

            if j - x_len + num_bins > i:
                row.append(None)

            elif j < i:
                row.append(None)

            elif j == i:
                prev = None
                if (i > 0):
                    prev = (i - 1, j - 1)
                row.append(opt_matrix_node((i, j), prev, 0))

            else:
                if 0 == i:
                    row.append(opt_matrix_node((i, j), None, sse_cache[j][j]))
                else:
                    potential_pos = []
                    potential_val = []
                    for k in range(i, j + 1):
                        sse_range = (k, j)
                        sse = sse_cache[j][j - k]
                        prev_pos = (i - 1, k - 1)
                        prev_opt = min_sse_mat[i - 1][k - 1]

                        potential_pos.append((sse_range, prev_pos))
                        potential_val.append(sse + prev_opt.val)
                    
                    # Find out the optimal answer in candidate answers
                    m = min(potential_val)
                    argmin = potential_val.index(m)
                    (sse_range, prev_pos) = potential_pos[argmin]

                    row.append(opt_matrix_node(sse_range, prev_pos, m))

        min_sse_mat.append(row)

    output_mat = []
    output_bins = []

    # Get the final answer from the last row of matrix
    min_answer = min_sse_mat[num_bins - 1][x_len - 1]
    # Loop from the last answer to the right beginning to
    # collect all the partitions
    while True:
        i, j = min_answer.sse_range
        output_bins.append(x[x_len - j - 1 : x_len - i])

        if min_answer.prev_pos == None:
            break
        i, j = min_answer.prev_pos
        min_answer = min_sse_mat[i][j]


    for i in range(num_bins):
        row = []
        for j in range(x_len):
            if min_sse_mat[i][x_len - j - 1] != None:
                row.append(min_sse_mat[i][x_len - j - 1].val)
            else:
                row.append(-1)
        output_mat.append(row)


    return output_mat, output_bins

