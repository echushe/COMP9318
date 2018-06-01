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



def generate_selected_dictionary(full_dict, size_limit):
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
    We assume that keys of the dictionary are already sorted here
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



def test_train(strategy_instance, parameters, features_max, full_dict, test_samples):

    print("======================================================================")
    print('Parameters: {}'.format(parameters))

    samples = combine_samples(strategy_instance.class0, strategy_instance.class1, test_samples)

    len_class_0 = len(strategy_instance.class0)
    len_class_1 = len(strategy_instance.class1)
    len_test = len(test_samples)

    for index in range(1, (features_max + 10) // 10):

        selected_dict = generate_selected_dictionary(full_dict, index * 10)

        train_test_data = generate_features(selected_dict, samples)

        # print(train_test_data)

        train_x = train_test_data[ : len_class_0 + len_class_1]
        train_y = [0.0 for i in range(len_class_0 + len_class_1)]
        for i in range(len_class_0, len_class_0 + len_class_1):
            train_y[i] = 1.0

        # Train SVM with features extracted from training samples
        try:
            model = strategy_instance.train_svm(parameters, train_x, train_y)
            # print(model.coef_)

            ############################### Re-test class 0 and class 1 of training set ##########################
            '''
            print(model.predict(train_x))
            print()
            '''
            ################################ test test_data.txt ##################################################

            # Predict the classification of the test samples
            test_x = train_test_data[len_class_0 + len_class_1 : ]
            prediction = model.predict(test_x)
            # print(prediction)
            sum = 0
            for val in prediction:
                if val > 0:
                    sum += 1

            print('Features used: {}\tPrediction Accuracy of test set: {}'.format(10 * index, sum / len(prediction)))
        except ValueError:
            print('ValueError')



def fool_classifier(test_data): ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...
    np.set_printoptions(threshold = np.nan)
    
    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
    strategy_instance = helper.strategy() 
    parameters = { 
        'C' : 10, 
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

    full_dict = generate_full_dictionary(strategy_instance.class0, strategy_instance.class1, test_samples)

    print(full_dict)
    print(len(full_dict))

    C_params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    degree_params = [1, 2, 3, 4, 5]
    gamma_params = ['auto', 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    coef0_params = [-100, -10, -5, 0, 5, 10, 100]

    max_features_allowed = 5000

    for kernel in ['linear']:#, 'poly', 'rbf', 'sigmoid', 'precomputed']:

        if kernel == 'linear':
            for C in C_params:
                
                parameters = { 
                    'C' : C, 
                    'kernel' : kernel,
                    'degree' : 3,
                    'gamma' : 'auto',
                    'coef0' : 0.0 }
                test_train(strategy_instance, parameters, max_features_allowed, full_dict, test_samples)

        elif kernel == 'poly':
            for C in C_params:
                for degree in degree_params:
                    for gamma in gamma_params:
                        for coef0 in coef0_params:

                            parameters = { 
                                'C' : C, 
                                'kernel' : kernel,
                                'degree' : degree,
                                'gamma' : gamma,
                                'coef0' : coef0 }
                            test_train(strategy_instance, parameters, max_features_allowed, full_dict, test_samples)

        elif kernel == 'rbf':
            for C in C_params:
                for gamma in gamma_params:

                    parameters = { 
                        'C' : C, 
                        'kernel' : kernel,
                        'degree' : 3,
                        'gamma' : gamma,
                        'coef0' : 0.0 }
                    test_train(strategy_instance, parameters, max_features_allowed, full_dict, test_samples)

        elif kernel == 'sigmoid':
            for C in C_params:
                for gamma in gamma_params:
                    for coef0 in coef0_params:

                        parameters = { 
                            'C' : C, 
                            'kernel' : kernel,
                            'degree' : 3,
                            'gamma' : gamma,
                            'coef0' : coef0 }
                        test_train(strategy_instance, parameters, max_features_allowed, full_dict, test_samples)

        elif kernel == 'precomputed':
            for C in C_params:
                parameters = { 
                    'C' : C, 
                    'kernel' : kernel,
                    'degree' : 3,
                    'gamma' : 'auto',
                    'coef0' : 0.0 }
                test_train(strategy_instance, parameters, max_features_allowed, full_dict, test_samples)
                        
    
    ## Write out the modified file, i.e., 'modified_data.txt' in Present Working Directory...
    
    
    ## You can check that the modified text is within the modification limits.
    modified_data='./modified_data.txt'
    assert strategy_instance.check_data(test_data, modified_data)

    return strategy_instance ## NOTE: You are required to return the instance of this class.


