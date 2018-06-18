import submission as submission
import pandas as pd

raw_data = pd.read_csv('spam.csv', encoding='latin-1')
raw_data.head()

def tokenize(sms):
    return sms.split(' ')

def get_freq_of_tokens(sms):
    tokens = {}
    for token in tokenize(sms):
        if token not in tokens:
            tokens[token] = 1
        else:
            tokens[token] += 1
    return tokens


for test_sms_index in range(len(raw_data)):
    
    training_data = []
    sms = None
    target = None
    predict = None

    for index in range(len(raw_data)):

        if index != test_sms_index:
            training_data.append((get_freq_of_tokens(raw_data.iloc[index].text), raw_data.iloc[index].category))
        else:
            sms = raw_data.iloc[index].text
            target = raw_data.iloc[index].category

    if submission.multinomial_nb(training_data, tokenize(sms)) > 1:
        predict = 'spam'
    else:
        predict = 'ham'

    print("target: {}\t Predict: {}\t {}".format(target, predict, target == predict))
