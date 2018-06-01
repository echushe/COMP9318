import helper
import numpy as np
import math
import operator
from sklearn.feature_extraction.text import TfidfVectorizer


def mod(vector):
    sum = 0.0
    for val in vector:
        sum += val * val

    return math.sqrt(sum)


def combine_samples(class_0, class_1, test_samples):
    """
    Combine all samples together
    """
    samples = []

    for i in range(len(class_0)):
        str = ''
        for word in class_0[i]:
            str += ' '
            str += word
        samples.append(str)

    for i in range(len(class_1)):
        str = ''
        for word in class_1[i]:
            str += ' '
            str += word
        samples.append(str)

    for i in range(len(test_samples)):
        str = ''
        for word in test_samples[i]:
            str += ' '
            str += word
        samples.append(str)


    return samples



def train(strategy_instance, parameters, test_samples):

    print("======================================================================")
    print('Parameters: {}'.format(parameters))

    samples = combine_samples(strategy_instance.class0, strategy_instance.class1, test_samples)

    # DO NOT forget positions of class_0, class_1 and test_data
    len_class_0 = len(strategy_instance.class0)
    len_class_1 = len(strategy_instance.class1)
    len_test = len(test_samples)

    vectorizer = TfidfVectorizer(analyzer="word", token_pattern='\S+') #, max_features = 500)
    train_x = vectorizer.fit_transform(samples[: len_class_0 + len_class_1])

    for i in range(len_class_0):
        train_x[i] *= len_class_1 / len_class_0

    train_y = np.zeros((len_class_0 + len_class_1))
    for i in range(len_class_0, len_class_0 + len_class_1):
        train_y[i] = 1.0

    # Train SVM with features extracted from training samples
    model = strategy_instance.train_svm(parameters, train_x, train_y)
    # print(model.coef_)

    # Test the train data to verify if the training is done (successful)
    train_y = model.predict(train_x)
    print(train_y)
    print()

    # Predict the classification of the test samples. 
    # the test samples are supposed to belong to class_1.
    test_x = vectorizer.transform(samples[len_class_0 + len_class_1 :])
    test_y = model.predict(test_x)
    print(test_y)
    sum = 0
    for val in test_y:
        if val > 0:
            sum += 1

    print('Features: {}'.format(test_x.shape[1]))
    print('Prediction Accuracy of test set: {}'.format(sum / len(test_y)))

    # print(test_x)


    return vectorizer, model, strategy_instance, samples, model.support_, model.support_vectors_, model.dual_coef_, train_x, train_y, test_x



def update_test_samples(vectorizer, samples, test_samples, weights, test_x):

    new_test_samples = []
    
    word_dict = vectorizer.vocabulary_
    invert_dict = dict((v, k) for k, v in word_dict.items())

    # print(word_dict)
    test_x = test_x.toarray()

    # Go through each vector
    for i in range(test_x.shape[0]):

        # This dictionary stores pairs of wj * xj and word index
        # If the word does not exist, stores wj
        index_w_map = {}
        for j in range(test_x.shape[1]):

            if test_x[i, j] > 0 and weights[0, j] > 0:
                index_w_map[j] = weights[0, j] * test_x[i, j]
            # If the feature == 0, which means this word does not exist in this sample
            # Insert (-1) * weight
            elif test_x[i, j] == 0 and weights[0, j] < 0:
                # The value should be >= 0 so we should multiply weight with -1
                index_w_map[j] = weights[0, j]
            elif test_x[i, j] > 0:
                index_w_map[j] = 0


        # Sort the dictionary by absolute value of wj * xj or (-1) * wj
        index_w_list = sorted(index_w_map.items(), key = operator.itemgetter(1))
        index_w_list.reverse()

        words = test_samples[i].split(' ')
        words_set = set(words)
        
        words_to_add = set([])
        words_to_remove = set([])

        # Update l_x_mul_w (add or remove words) until it exceeds target_pos
        for j in range(20):
            index, w = index_w_list[j]
            word = invert_dict[index]

            if weights[0, index] > 0:

                if word in words_set:
                    words_to_remove.add(word)
                else:
                    words_to_add.add(word)
            
            else:
                if word in words_set:
                    words_to_remove.add(word)
                else:
                    words_to_add.add(word)


        new_test_samples.append([])

        # print(words_to_remove)

        for word in words:
            if word not in words_to_remove:
                new_test_samples[-1].append(word)

        for word in words_to_add:
            new_test_samples[-1].append(word)

        print('Words updated in this line: {} + {} = {}'.format(len(words_to_remove),
                                                                len(words_to_add),
                                                                len(words_to_remove) + len(words_to_add)))

    out = ''
    for sample in new_test_samples:
        for word in sample:
            out += word
            out += ' '
        out += '\n'

    with open("./modified_data.txt", "w") as text_file:
        text_file.write(out)



def investigate_train_results( param ):

    vectorizer, model, strategy_instance, samples, sv_indices, sv_vectors, dual_coef, train_x, train_y, test_x = param
    
    # DO NOT forget positions of class_0, class_1 and test_data
    len_class_0 = len(strategy_instance.class0)
    len_class_1 = len(strategy_instance.class1)
    len_test = test_x.shape[0]
    '''
    print(dual_coef)
    print()
    print()
    print(sv_vectors)
    print()
    print()
    print(sv_indices)
    '''
    # print(weights.shape)
    # print(weights)

    # Shape of weights: [1, n_features]
    # dual_coef = dual_coef.todense()
    # sv_vectors = sv_vectors.todense()
    weights = np.dot(dual_coef, sv_vectors)

    x_mul_w = np.dot(train_x, np.transpose(weights))
    t_x_mul_w = np.dot(test_x, np.transpose(weights))

    boundary_up = 0.0
    n_up = 0

    boundary_down = 0.0
    n_down = 0

    for i in range(len(sv_indices)):

        if sv_indices[i] < len_class_0:
            boundary_up += x_mul_w[sv_indices[i], 0]
            n_up += 1
            # print(x_mul_w[sv[i]])

        elif sv_indices[i] >= len_class_0:
            boundary_down += x_mul_w[sv_indices[i], 0]
            n_down += 1
            # print(x_mul_w[sv[i]])

    boundary_up /= n_up
    boundary_down /= n_down
    boundary = (boundary_up + boundary_down) / 2

    margin = boundary_down - boundary_up

    print('bias: {}'.format(model.intercept_))
    print('boundary: {}'.format(boundary))
    print('Up boundary: {}'.format(boundary_up))
    print('Down boundary: {}'.format(boundary_down))


    test_samples = samples[len_class_0 + len_class_1 : ]
    update_test_samples(vectorizer, samples, test_samples, weights, test_x)

    with open('./modified_data.txt','r') as test_text:
        modified_test_samples = [line for line in test_text]

    modified_test_x = vectorizer.transform(modified_test_samples)
    print(model.predict(modified_test_x))

    modified_x_mul_w = np.dot(modified_test_x, np.transpose(weights))
    x_mul_w_pairs = []
    for i in range(modified_x_mul_w.shape[0]):
        x_mul_w_pairs.append((t_x_mul_w[i, 0], modified_x_mul_w[i, 0]))

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
    
    parameters = { 
        'C' : 1, 
        'kernel' : 'linear',
        'degree' : 3,
        'gamma' : 'auto',
        'coef0' : 1 }

    investigate_train_results(train(strategy_instance, parameters, test_samples))
    ## Write out the modified file, i.e., 'modified_data.txt' in Present Working Directory...
    
    
    ## You can check that the modified text is within the modification limits.
    modified_data='./modified_data.txt'
    assert strategy_instance.check_data(test_data, modified_data)
    

    return strategy_instance ## NOTE: You are required to return the instance of this class.

