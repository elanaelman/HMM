# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 17:43:19 2022

@author: Elana
"""

import argparse
from hmm import *

def main():
    parser = argparse.ArgumentParser(description='Program to build and train a neural network.')
    parser.add_argument('--hmm', default=None, help='Path to the trained hmm.')
    parser.add_argument('--test_data', default=None, help='Path to the testing data.')
    parser.add_argument('--t', default=10, help='Number of characters before prediction')
    args = parser.parse_args()
    
    hmm = load_hmm(args.hmm)
    
    correct = 0
    total = 0
    samples = format_dataset(load_subdir(args.test_data))
    time = 10
    steps = 2
    
    for sample in samples:
        if len(sample) >= time + steps:
            short_sample = sample[:time]
            prediction = hmm.complete_sequence(short_sample, steps)
            
            total += 1
            if [sample[time+step] == prediction[step] for step in range(steps)].all():
                correct += 1
    accuracy = correct/total
    print(f"Probability of correctly guessing character {time} based on the preceeding string is {accuracy}%.")
    
    
if __name__ == '__main__':
    main()