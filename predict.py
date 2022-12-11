# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 17:43:19 2022

@author: Elana
"""

import argparse
from hmm import *


def predict_run(trained_hmm = None, test_data = None, sample_size = 10, time=10, steps=1):

    hmm = trained_hmm
    if __name__ == '__main__':
        args = predict_args()
        hmm = load_hmm(args.hmm)
        data = args.test_data
        time = args.t
        steps = args.d
    else:
        hmm = load_hmm(trained_hmm)
        data = test_data

    correct = 0
    total = 0
    samples = pick_data(format_dataset(load_subdir(data)), sample_size)
    
    for sample in samples:
        if len(sample) > time + steps:
            prediction = hmm.complete_sequence(sample[:time], steps)
            total += 1
            if [sample[time+s] == prediction[s] for s in range(steps)].all():
                correct += 1
    accuracy = correct/total*100
    print(f"Probability of correctly guessing character {time} based on the preceeding string is {accuracy}%.")


def predict_args():
    parser = argparse.ArgumentParser(description='Program to build and train a neural network.')
    parser.add_argument('--hmm', default=None, help='Path to the trained hmm.')
    parser.add_argument('--test_data', default=None, help='Path to the testing data.')
    parser.add_argument('--t', default=10, help='Number of characters before prediction')
    parser.add_argument('--d', default=1, help="number of characters to predict")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    predict_run()