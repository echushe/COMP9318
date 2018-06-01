import helper
import numpy as np
import math
import operator


def generate_full_dictionary(samples, len_class_0, len_class_1):
    """
    This function is to generate a full dictionary including all words
    from both training set and test set
    """
    dict = {}

    for i in range(len_class_0 + len_class_1):
        for word in samples[i]:

            if word not in dict:
                dict[word] = 1
            else:
                dict[word] += 1

    return dict



def generate_selected_dictionary(full_dict, size_limit):
    """
    Generate a shorter dictionary including most frequently used words
    """
    dict_revised = {}

    sorted_list = sorted(full_dict.items(), key = operator.itemgetter(1))
    sorted_list.reverse()

    begin = 0
    for i in range(begin, min(size_limit, len(full_dict))):
        key, value = sorted_list[i]
        dict_revised[key] = value

    return dict_revised



def mod(vector):
    sum = 0.0
    for val in vector:
        sum += val * val

    return math.sqrt(sum)


def generate_feature(dict, sample):
    """
    Generate a vector of float numbers from a sample.
    Size of this vector is equal to size of the dictionary.
    """
    vector = np.zeros(len(dict), dtype = float)
    sample_dict = {}

    for word in sample:
        if word not in sample_dict:
            sample_dict[word] = 1
        else:
            sample_dict[word] += 1

    i = 0
    for key, value in dict.items():
        if key in sample_dict:         
            vector[i] = sample_dict[key]
        i += 1

    # vector *= mod(vector)

    return vector



def combine_samples(class_0, class_1, test_samples):
    """
    Combine all samples together
    """
    samples = []
    samples.extend(class_0)
    samples.extend(class_1)
    samples.extend(test_samples)

    return samples



def generate_features(dict, samples):
    """
    Input is a list of samples.
    Each sample is a list of words.
    The generated data is a 2D numpy array.
    """

    train_test_data = np.zeros((len(samples), len(dict)), dtype=float)

    for i in range(len(samples)):
        train_test_data[i] = generate_feature(dict, samples[i])

    return train_test_data



def train(strategy_instance, parameters, features_used, test_samples):

    print("======================================================================")
    print('Parameters: {}'.format(parameters))

    samples = combine_samples(strategy_instance.class0, strategy_instance.class1, test_samples)

    # DO NOT forget positions of class_0, class_1 and test_data
    len_class_0 = len(strategy_instance.class0)
    len_class_1 = len(strategy_instance.class1)
    len_test = len(test_samples)

    full_dict = generate_full_dictionary(samples, len_class_0, len_class_1)

    selected_dict = generate_selected_dictionary(full_dict, features_used)

    features = generate_features(selected_dict, samples)

    # Get training set via its start and end positions
    train_x1 = np.zeros((len_class_0 // 2 + len_class_1, features.shape[1]), dtype = float)
    for i in range(len_class_0 // 2):
        train_x1[i] = features[i]
    for i in range(len_class_0, len_class_0 + len_class_1):
        train_x1[i - (len_class_0 - len_class_0 // 2)] = features[i]

    train_x2 = features[len_class_0 // 2 : len_class_0 + len_class_1]


    for i in range(len_class_0 // 2):
        train_x1[i] *= len_class_1 / len_class_0

    for i in range(len_class_0 - len_class_0 // 2):
        train_x2[i] *= len_class_1 / len_class_0


    train_y1 = [0.0 for i in range(len_class_0 // 2 + len_class_1)]
    train_y2 = [0.0 for i in range(len_class_0 - len_class_0 // 2 + len_class_1)]
    for i in range(len_class_0 // 2, len_class_0 // 2 + len_class_1):
        train_y1[i] = 1.0
    for i in range(len_class_0 - len_class_0 // 2, len_class_0 - len_class_0 // 2 + len_class_1):
        train_y2[i] = 1.0

    # Train SVM with features extracted from training samples
    model1 = strategy_instance.train_svm(parameters, train_x1, train_y1)
    model2 = strategy_instance.train_svm(parameters, train_x2, train_y2)
    # print(model.coef_)

    # Test the train data to verify if the training is done (successful)
    print(model1.predict(train_x1))
    print(model2.predict(train_x2))
    # print(train_y)
    print()

    # Predict the classification of the test samples. 
    # the test samples are supposed to belong to class_1.
    test_x = features[len_class_0 + len_class_1 : ]
    test_y = model1.predict(test_x)
    print(test_y)

    sum = 0
    for val in test_y:
        if val > 0:
            sum += 1

    print('Features: {}'.format(test_x.shape[1]))
    print('Prediction Accuracy of test set: {}'.format(sum / len(test_y)))

    test_y = model2.predict(test_x)
    print(test_y)
    
    sum = 0
    for val in test_y:
        if val > 0:
            sum += 1

    print('Features: {}'.format(test_x.shape[1]))
    print('Prediction Accuracy of test set: {}'.format(sum / len(test_y)))

    # print(test_x)


    return model1, model2, selected_dict, samples, features, len_class_0, len_class_1, train_x1, train_y1, train_x2, train_y2, test_x, test_y


def get_key(param1, param2):
    val1, val2 = param2
    return val1


def update_test_samples(selected_dict, test_samples, weights, test_x, test_y):
    
    word_occ_pair_list = list(selected_dict.items())

    new_test_samples = []

    # Go through each vector
    for i in range(len(test_x)):

        # This dictionary stores pairs of wj * xj and word index
        # If the word does not exist, stores wj
        index_w_map = {}
        for j in range(len(test_x[i])):
            
            if test_x[i, j] > 0 and weights[0, j] > 0:
                index_w_map[j] = weights[0, j] * test_x[i, j]
            # If the feature == 0, which means this word does not exist in this sample
            # Insert (-1) * weight
            elif test_x[i, j] == 0 and weights[0, j] < 0:
                # The value should be >= 0 so we should multiply weight with -1
                index_w_map[j] = weights[0, j]


        # Sort the dictionary by absolute value of wj * xj or (-1) * wj
        index_w_list = sorted(index_w_map.items(), key = operator.itemgetter(1))
        index_w_list.reverse()

        # index_w_list = list(index_w_map.items())
        
        words_to_add = set([])
        words_to_remove = set([])

        # Update l_x_mul_w (add or remove words) until it exceeds target_pos
        for j in range(20):
            index, w = index_w_list[j]
            word, occ = word_occ_pair_list[index]

            if weights[0, index] > 0:

                words_to_remove.add(word)
            
            else:                
                words_to_add.add(word)


        new_test_samples.append([])

        for word in test_samples[i]:
            if word not in words_to_remove:
                new_test_samples[-1].append(word)

        for word in words_to_add:
            new_test_samples[-1].append(word)

        print('Words updated in this line: {} + {}'.format(len(words_to_remove), len(words_to_add)))

    out = ''
    for sample in new_test_samples:
        for word in sample:
            out += word
            out += ' '
        out += '\n'

    with open("./modified_data_dual_svm.txt", "w") as text_file:
        text_file.write(out)


def get_up_down_boundaries(model, train_x, train_y):
    sv = model.support_
    w = model.coef_

    # print(sv)
    # print(weights.shape)
    # print(weights)

    x_mul_w = np.matmul(train_x, np.transpose(w))

    boundary_up = 0.0
    n_up = 0

    boundary_down = 0.0
    n_down = 0

    for i in range(len(sv)):

        if sv[i] < len(train_x) // 2 and train_y[sv[i]] == 0:
            boundary_up += x_mul_w[sv[i], 0]
            n_up += 1
            # print(x_mul_w[sv[i]])

        elif sv[i] >= len(train_x) // 2 and train_y[sv[i]] == 1:
            boundary_down += x_mul_w[sv[i], 0]
            n_down += 1
            # print(x_mul_w[sv[i]])

    boundary_up /= n_up
    boundary_down /= n_down

    print('Bias: {}'.format(model.intercept_))
    print('Up boundary: {}'.format(boundary_up))
    print('Down boundary: {}'.format(boundary_down))

    return boundary_up, boundary_down




def investigate_train_results( param ):

    model1, model2, selected_dict, samples, features, len_class_0, len_class_1, train_x1, train_y1, train_x2, train_y2, test_x, test_y = param
    
    get_up_down_boundaries(model1, train_x1, train_y1)
    get_up_down_boundaries(model2, train_x2, train_y2)

    ave_weights = (model1.coef_ + model2.coef_) / 2

    test_samples = samples[len_class_0 + len_class_1 : ]
    update_test_samples(selected_dict, test_samples, ave_weights, test_x, test_y)

    with open('./modified_data_dual_svm.txt','r') as test_text:
        modified_test_samples = [line.strip().split(' ') for line in test_text]

    modified_test_x = generate_features(selected_dict, modified_test_samples)
    print(model1.predict(modified_test_x))
    print(model2.predict(modified_test_x))

    x_mul_w = np.matmul(test_x, np.transpose(ave_weights))
    modified_x_mul_w = np.matmul(modified_test_x, np.transpose(ave_weights))
    x_mul_w_pairs = []
    for i in range(len(modified_x_mul_w)):
        x_mul_w_pairs.append((x_mul_w[i, 0], modified_x_mul_w[i, 0]))

    for v1, v2 in x_mul_w_pairs:
        print('before: {}\tafter: {}'.format(v1, v2))



def fool_classifier(test_data): ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...
    np.set_printoptions(threshold = np.nan)
    
    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
    strategy_instance = helper.strategy() 

    with open(test_data,'r') as test_text:
        test_samples = [line.strip().split(' ') for line in test_text]

    class_0 = strategy_instance.class0
    class_1 = strategy_instance.class1

    # print(full_dict)
    # print(len(full_dict))
    '''
    for C in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        for i in range(5, 61):
            parameters = { 
                'C' : C, 
                'kernel' : 'linear',
                'degree' : 3,
                'gamma' : 'auto',
                'coef0' : 0.0 }

            train(strategy_instance, parameters, 10 * i, test_samples)
    '''
    
    parameters = { 
        'C' : 1, 
        'kernel' : 'linear',
        'degree' : 3,
        'gamma' : 'auto',
        'coef0' : 0.0 }

    ## train(strategy_instance, parameters, 10000, test_samples)
    investigate_train_results(train(strategy_instance, parameters, 10000, test_samples))
    ## Write out the modified file, i.e., 'modified_data.txt' in Present Working Directory...
    
    
    ## You can check that the modified text is within the modification limits.
    modified_data='./modified_data_dual_svm.txt'
    assert strategy_instance.check_data(test_data, modified_data)
    

    return strategy_instance ## NOTE: You are required to return the instance of this class.


