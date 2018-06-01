import helper
import numpy as np
import operator
import math


def generate_full_dictionary(class_0, class_1, test):
    dict = {}

    for line in class_0:
        for word in line:

            if word not in dict:
                dict[word] = 1
            else:
                dict[word] += 1

    for line in class_1:
        for word in line:

            if word not in dict:
                dict[word] = 1
            else:
                dict[word] += 1

    for line in test:
        for word in line:

            if word not in dict:
                dict[word] = 1
            else:
                dict[word] += 1


    return dict



def generate_dictionary_for_co_occurrence_matrix(full_dict, size_limit):
    dict_revised = {}

    sorted_list = sorted(full_dict.items(), key = operator.itemgetter(1))
    sorted_list.reverse()

    begin = 0
    end = size_limit
    for i in range(begin, min(size_limit, len(full_dict))):
        key, value = sorted_list[i]
        # Values in the revised dictionary are index values
        dict_revised[key] = i - begin

    return dict_revised




def combine_samples(class_0, class_1, test_samples):
    samples = []
    samples.extend(class_0)
    samples.extend(class_1)
    samples.extend(test_samples)

    return samples



def mod(vector):
    sum = 0.0
    for val in vector:
        sum += val * val

    return math.sqrt(sum)



def generate_co_occurrence_matrix(dict, samples):

    print ('generate_co_occurrence_matrix ...')

    word_list = dict.keys();

    matrix = np.zeros((len(dict), len(dict)), dtype = float)

    for sample in samples:

        for i in range(len(sample)):
            for j in range(len(sample)):
                if i != j:
                    if (sample[i] in dict) and (sample[j] in dict):
                        row = dict[sample[i]]
                        col = dict[sample[j]]
                        matrix[row, col] += 1

    print ('generate_co_occurrence_matrix ... finish')
    return matrix



def generate_doc_vec_via_word_vec(dict, word_vecs, samples):

    train_test_data = np.zeros((len(samples), word_vecs.shape[1]), dtype = float)

    # train_test_data = np.full((len(samples), word_vecs.shape[1]), 1, dtype = float)
    log_v = np.vectorize(np.math.log)

    for i in range(len(samples)):
        for word in samples[i]:
            if word in dict:
                train_test_data[i] += word_vecs[dict[ word ]]

    return train_test_data



def my_svd(matrix, features):
    
    U, s, VT = np.linalg.svd(matrix, full_matrices = False)

    V = np.transpose(VT)

    if features < U.shape[1]:
        new_U = np.zeros((matrix.shape[0], features))
        new_V = np.zeros((matrix.shape[0], features))
        new_S = np.copy(s[ : features])

        for i in range(len(new_U)):
            new_U [i] = U[i, : features]
            new_V [i] = V[i, : features]

        return new_U, new_S, new_V

    else:
        return U, s, V


def add_weight_to_word_vectors(vectors, s):
    for i in range(len(vectors)):
        vectors[i] = np.dot(vectors[i], s)


def fool_classifier(test_data): ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...

    np.set_printoptions(threshold = np.nan)
    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
    strategy_instance = helper.strategy() 
    parameters = { 
        'C' : 1, 
        'kernel' : 'linear',
        'degree' : 2,
        'gamma' : 'auto',
        'coef0' : 0.0 }
    '''
        gamma=parameters['gamma']
        C=parameters['C']
        kernel=parameters['kernel']
        degree=parameters['degree']
        coef0=parameters['coef0']
    '''
    with open(test_data,'r') as test_text:
        test_samples = [line.strip().split(' ') for line in test_text]

    class_0 = strategy_instance.class0
    class_1 = strategy_instance.class1

    full_dict = generate_full_dictionary(class_0, class_1, test_samples)

    print(full_dict)
    print(len(full_dict))

    selected_dict = generate_dictionary_for_co_occurrence_matrix(full_dict, 2000)

    samples = combine_samples(class_0, class_1, test_samples)

    matrix = generate_co_occurrence_matrix(selected_dict, samples)

    # print (matrix)
    for index in range(1, 201):
        U, s, V = my_svd(matrix, 10 * index)
        '''
        print (U[0])
        print()
        print (s)
        '''
        # add_weight_to_word_vectors(U, s)

        train_test_data = generate_doc_vec_via_word_vec(selected_dict, U, samples)

        train_x = train_test_data[ : len(class_0) + len(class_1)]
        train_y = [0.0 for i in range(len(class_0) + len(class_1))]
        for i in range(len(class_0), len(class_0) + len(class_1)):
            train_y[i] = 1.0

        # Train SVM with features extracted from training samples
        model = strategy_instance.train_svm(parameters, train_x, train_y)

        ################################ Re-test class 0 of training set #####################################
        '''
        # Predict the classification of the test samples
        test_x = train_test_data[ : len(class_0)]
        print(model.predict(test_x))
        print()
        '''
        ################################ Re-test class 1 of training set #####################################
        '''
        # Predict the classification of the test samples
        test_x = train_test_data[len(class_0) : len(class_0) + len(class_1)]
        print(model.predict(test_x))
        print()
        '''
        ############################### Re-test class 0 and class 1 of training set ##########################
        '''
        print(model.predict(train_x))
        print()
        '''
        ################################ test test_data.txt ##################################################

        # Predict the classification of the test samples
        test_x = train_test_data[len(class_0) + len(class_1) : ]
        prediction = model.predict(test_x)
        # 
        # 
        # 
        # print(prediction)
        sum = 0
        for val in prediction:
            if val > 0:
                sum += 1

        print('Features used: {}'.format(10 * index))
        print('Prediction Accuracy of test set: {}'.format(sum / len(prediction)))
    
    ## Write out the modified file, i.e., 'modified_data.txt' in Present Working Directory...
    
    
    ## You can check that the modified text is within the modification limits.
    modified_data='./modified_data.txt'
    assert strategy_instance.check_data(test_data, modified_data)

    return strategy_instance ## NOTE: You are required to return the instance of this class.

