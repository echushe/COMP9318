## import modules here 
import math

################# Question 1 #################
def generate_rate_dict(training_data):
    
    # define variables to return
    vocabulary = set([])

    spam_dict = {}
    n_spam = 0
    n_spam_sms = 0

    ham_dict = {}
    n_ham = 0
    n_ham_sms = 0

    for (dict, category) in training_data:

        for key in dict:
            vocabulary.add(key)

        if category == 'spam':
            n_spam_sms += 1

            for key, value in dict.items():
                
                n_spam += value
                
                if key not in spam_dict:
                    spam_dict[key] = value
                else:
                    spam_dict[key] += value


        elif category == 'ham':
            n_ham_sms += 1

            for key, value in dict.items():

                n_ham += value

                if key not in ham_dict:
                    ham_dict[key] = value
                else:
                    ham_dict[key] += value

    return spam_dict, n_spam, n_spam_sms, ham_dict, n_ham, n_ham_sms, vocabulary



def multinomial_nb(training_data, sms):# do not change the heading of the function

    '''
    for (dict, category) in training_data:
        print('-----------------------------------------------------------')
        print(dict)
        print(category)
    '''


    spam_dict, n_spam, n_spam_sms, ham_dict, n_ham, n_ham_sms, vocabulary = generate_rate_dict(training_data)

    p_spam = math.log(n_spam_sms);
    p_ham = math.log(n_ham_sms);

    #print("n_spam_sms: {}\tn_ham_sms: {}".format(n_spam_sms, n_ham_sms))
    #print("n_spam: {}\tn_ham: {}\tvocabulary: {}".format(n_spam, n_ham, len(vocabulary)))

    divider_spam = n_spam + len(vocabulary);
    divider_ham = n_ham + len(vocabulary);

    for word in sms:

        if word not in vocabulary:
            p_spam += 0
        elif word not in spam_dict:
            p_spam += math.log(1 / divider_spam)
        else:
            p_spam += math.log((spam_dict[word] + 1) / divider_spam)

        if word not in vocabulary:
            p_ham += 0
        elif word not in ham_dict:
            p_ham += math.log(1 / divider_ham)
        else:
            p_ham += math.log((ham_dict[word] + 1) / divider_ham)

    #print("p_spam: {}\tp_ham: {}".format(p_spam, p_ham))

    return math.exp(p_spam) / math.exp(p_ham)