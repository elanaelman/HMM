# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 17:43:19 2022

@author: Elana
"""

import argparse
from hmm import *


def predict_run(trained_hmm = None, test_data = None, sample_size = 10, time=10):

    hmm = trained_hmm
    if __name__ == '__main__':
        args = predict_args()
        hmm = load_hmm(args.hmm)
        data = args.test_data
        time = args.t
    else:
        hmm = load_hmm(trained_hmm)
        data = test_data


    samples = pick_data(format_dataset(load_subdir(data)), sample_size)
    result = []
    for sample in samples:
        '''
        if len(sample) > time + steps:
            prediction = hmm.complete_sequence(sample[:time], steps)
            total += 1
            if np.array([sample[time+s] == prediction[s] for s in range(steps)]).all():
                correct += 1
        '''
        result.append(last_good_prediction(hmm, sample, time))
    
    return result
    #print(f"Probability of correctly guessing {steps} characters after the {time}th based on the preceeding string is {accuracy}.")

def last_good_prediction(hmm, sample, t):
    steps = 1
    while len(sample) > t + steps:
        prediction = hmm.complete_sequence(sample[:t+steps], 1)
        if sample[t+steps] != prediction:
            break
        else:
            steps += 1
    return steps-1
        
        

def predict_args():
    parser = argparse.ArgumentParser(description='Program to build and train a neural network.')
    parser.add_argument('--hmm', default=None, help='Path to the trained hmm.')
    parser.add_argument('--test_data', default=None, help='Path to the testing data.')
    parser.add_argument('--t', default=10, help='Number of characters before prediction')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    predict_run()