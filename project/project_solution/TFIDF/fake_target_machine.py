import helper
import numpy as np
import math
import operator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

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
    train_x = vectorizer.fit_transform(samples)
    '''
    for i in range(len_class_0):
        train_x[i] *= len_class_1 / len_class_0
    '''
    train_y = np.zeros(len(samples))
    for i in range(len_class_0, len(samples)):
        train_y[i] = 1.0

    # Train SVM with features extracted from training samples
    ## Populate the parameters...
    gamma=parameters['gamma']
    C=parameters['C']
    kernel=parameters['kernel']
    degree=parameters['degree']
    coef0=parameters['coef0']
        
    ## Train the classifier...
    model = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, coef0=coef0)
    model.fit(train_x, train_y)
    # print(model.coef_)

    # Test the train data to verify if the training is done (successful)
    train_y = model.predict(train_x)
    print(train_y)
    print()


    return vectorizer, model




def investigate_train_results( param, modified ):

    vectorizer, model = param

    with open(modified,'r') as test_text:
        modified_test_samples = [line for line in test_text]

    modified_test_x = vectorizer.transform(modified_test_samples)
    modified_test_y = model.predict(modified_test_x)

    total = 0
    zeros = 0
    for ele in modified_test_y:
        total += 1
        if ele == 0:
            zeros += 1

    print(modified_test_y)
    print(zeros / total)



def train_target_and_validate_changed_txt(test_file, modified): ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...
    np.set_printoptions(threshold = np.nan)
    
    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
    strategy_instance = helper.strategy()

    with open(test_file,'r') as test_text:
        test_samples = [line.strip().split(' ') for line in test_text]
    
    parameters = { 
        'C' : 1, 
        'kernel' : 'linear',
        'degree' : 3,
        'gamma' : 'auto',
        'coef0' : 1 }

    investigate_train_results(train(strategy_instance, parameters, test_samples), modified)
    ## Write out the modified file, i.e., 'modified_data.txt' in Present Working Directory...
    

    return strategy_instance ## NOTE: You are required to return the instance of this class.


