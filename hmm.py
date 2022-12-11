# File: hmm.py
# Purpose:  Starter code for building and training an HMM in CSC 246.

# Code by Elana Elman and Henry Lin
# References: "A Revealing Introduction to Hidden Markov Models" by Mark Stamp; PRML

import argparse
import os

import numpy
import numpy as np
from math import inf
import time

import pickle


# A utility class for bundling together relevant parameters - you may modify if you like.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# num_states -- this should be an integer recording the number of hidden states
#
# pi -- this should be the distribution over the first hidden state of a sequence
#
# transitions -- this should be a num_states x num_states matrix of transition probabilities
#
# emissions -- this should be a num_states x vocab_size matrix of emission probabilities
#              (i.e., the probability of generating token X when operating in state K)
#
# vocab_size -- this should be an integer recording the vocabulary size; 255 is a safe upper bound
#
# Note: You may want to add fields for expectations.
class HMM:
    __slots__ = ('pi', 'transitions', 'emissions', 'num_states', 'vocab_size', 'states')

    # The constructor should initalize all the model parameters.
    # you may want to write a helper method to initialize the emission probabilities.
    def __init__(self, num_states=10, vocab_size=255):
        self.num_states = num_states
        self.vocab_size = vocab_size
        self.transitions = self.random_matrix((self.num_states, self.num_states))
        self.emissions = self.random_matrix((self.num_states, self.vocab_size))
        self.pi = self.random_array(self.num_states)
        self.states = np.arange(num_states)

    # return the avg loglikelihood for a complete dataset (train OR test) (list of arrays)
    #TODO: combine with LL_helper for speed?
    def LL(self, dataset):
        # apply LL_helper to each sample in dataset. Return average.
        return np.average([self.LL_helper(sample) for sample in dataset])

    # return the LL for a single sequence (numpy array)
    def LL_helper(self, sample):
        c = self.alpha_pass(sample)[1]
        return -np.sum(np.log(c))

    # return the most likely state at time t based on a sequence of observations
    def guess_state(self, sample, t):
        alpha, beta, c, gamma, di_gamma = self.estimate(sample)
        return np.argmax(gamma[t])

    # helper method to randomize 2D matrix of probabilities
    def random_matrix(self, shape):
        matrix = np.random.normal(10, 0.1, shape)
        for i in range(shape[0]):
            matrix[i] = matrix[i] / np.sum(matrix[i])
        return matrix

    # helper method to randomize a probability distribution
    def random_array(self, length):
        arr = np.random.normal(10, 0.1, length)
        arr = arr / np.sum(arr)
        return arr

    # helper method to normalize a vector
    def normalize(self, vec):
        return vec / np.sum(vec)
    
    def alpha_pass(self, sample):
        T = len(sample)
        c = np.zeros((T))
        alpha = np.zeros((T, self.num_states))
        
        alpha[0] = np.vectorize(lambda i: self.pi[i] * self.emissions[i, sample[0]])(self.states)
        c[0] = 1 / np.sum(alpha[0])
        alpha[0] = alpha[0] * c[0]
        
        #alpha_new = alpha.copy()
        #c_new = c.copy()
        
        for t in range(1, T):
            for i in self.states:
                alpha[t, i] = np.sum(np.vectorize(lambda j: alpha[t - 1, j] * self.transitions[j, i])(self.states))*self.emissions[i, sample[t]]
            #f = lambda i, j: alpha_new[t - 1, j] * self.transitions[j, i]
            #g = lambda i: np.sum(f(i, self.states))*self.emissions[i, sample[t]]
            #alpha_new[t] = g(self.states)
            #c_new[t] = 1/np.sum(alpha_new[t])
            #alpha_new[t] *= c_new[t]
            c[t] = 1 / np.sum(alpha[t])
            alpha[t] = c[t] * alpha[t]
        
        return alpha, c
    
    def beta_pass(self, sample, c):
        T = len(sample)
        beta = np.zeros((T, self.num_states))
        beta[T - 1] = c[T - 1]
        for t in range(T - 2, -1, -1):
            for i in self.states:
                beta[t] = np.sum(
                    [self.transitions[i, j] * self.emissions[j, sample[t + 1]] * beta[t + 1, j] * c[t] for j in
                     self.states])
        return beta
    
    def calc_gammas(self, sample, alpha, beta):
        T = len(sample)
        gamma = np.zeros((T, self.num_states))
        di_gamma = np.zeros((T, self.num_states, self.num_states))
    
        # calculate gammas:
        for t in range(T - 1):
            for i in self.states:
                for j in self.states:
                    di_gamma[t, i, j] = alpha[t, i] * self.transitions[i, j] * self.emissions[
                        j, sample[t + 1]] * beta[t + 1, j]
                gamma[t, i] = np.sum(di_gamma[t, i])
        gamma[T - 1] = alpha[T - 1]
        return gamma, di_gamma
        
        
        
    
    def estimate(self, sample):
        # alpha[t, i] is the probability of the partial observation sequence up to time t,
        # such that the model is in hidden state i at time t.
        # beta[t, i] is the probability of observations after time t,
        # given that the model is in state i at time t.
        # gamma[t, i] is the probability of being in state i at time t, given the observations.
        # di_gamma[t, i, j] is the probability of being in state i at time t and then
        # transitioning to state j at time t+1, given a sequence of observations.
        
        #start_time = time.time()
        alpha, c = self.alpha_pass(sample)
        #after_alpha = time.time()
        #print(f'\tAlpha took {after_alpha-start_time} s')
        beta = self.beta_pass(sample, c)
        #after_beta = time.time()
        #print(f'\tBeta took {after_beta-after_alpha} s')
        gamma, di_gamma = self.calc_gammas(sample, alpha, beta)
        #after_gamma = time.time()
        #print(f'\tGamma took {after_gamma-after_beta} s')
        return alpha, beta, c, gamma, di_gamma

    def em_step(self, dataset):
        
        transition_numerators = np.zeros((self.num_states, self.num_states))
        transition_denominators = np.zeros((self.num_states))
        emission_numerators = np.zeros((self.num_states, self.vocab_size))
        emission_denominators = np.zeros((self.num_states))
        gamma_0 = np.zeros((self.num_states))
        sum_LL = 0

        for sample in dataset:   
            alpha, beta, c, gamma, di_gamma = self.estimate(sample)

            gamma_0 += gamma[0]
            for i in self.states:
                for t in range(len(sample) - 1):
                    transition_denominators[i] += gamma[t, i]
                    for j in self.states:
                        transition_numerators[i, j] += di_gamma[t, i, j]

            for j in self.states:
                for k in range(self.vocab_size):
                    emission_numerators[j, k] += np.sum(gamma[np.nonzero(sample == k), j])
                emission_denominators[j] += np.sum(gamma[:, j])

            sum_LL += np.sum(np.log(c))

        # combine sample data
        self.pi = self.normalize(gamma_0)
        for i in self.states:
            for j in self.states:
                self.transitions[i, j] = transition_numerators[i, j] / transition_denominators[i]
                self.emissions[i, j] = emission_numerators[i, j] / emission_denominators[i]

        newLL = -sum_LL / len(dataset)
        return newLL
        
        
    def train(self, train_data, maxIters):
        oldLL = -inf
        
        timer = time.time()
        
        for m in range(maxIters):
            newLL = self.em_step(train_data)
            new_time = time.time()
            print(f'iteration {m} took {new_time - timer} seconds')
            timer = new_time
            
            print(newLL)
            if newLL >= oldLL:
                oldLL = newLL
            else:
                print(f'Log likelihood decreased on iteration {m}.')
                break
        

    # Return a "completed" sample by additing additional steps based on model probability.
    def complete_sequence(self, sample, steps):
        T = len(sample)
        alpha, c = self.alpha_pass(sample)
        prevState = np.argmax(alpha[T-1])
        additions = []
        for t in range(steps):
            state = np.random.choice(self.num_states, p=self.transitions[prevState])
            character = np.random.choice(self.vocab_size, p=self.emissions[state])
            additions.append(character)
            prevState = state
        return additions

    # Save the complete model to a file (most likely using np.save and pickles)
    def save_model(self, filename):
        model = (self.transitions, self.emissions, self.pi)
        # np.save('model', model)
        # np.savetxt("file.csv", model, delimiter=",")
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print("Saved to " + filename)


# Load a complete model from a file and return an HMM object (most likely using np.load and pickles)
def load_hmm(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    ret = HMM(num_states=len(model[2]))
    ret.transitions = model[0]
    ret.emissions = model[1]
    ret.pi = model[2]
    print('Loaded from ' + filename)
    return ret


# Load all the files in a subdirectory and return a giant list.
def load_subdir(path):
    data = []
    for filename in os.listdir(path):
        with open(os.path.join(path, filename)) as fh:
            data.append(fh.read())
    return data

def load_sample(path):
    data = []
    with open(path) as file:
        data.append(file.read())
    return data

def pick_data(dataset, count):
    idx = np.random.choice(np.arange(len(dataset)), count)
    ret = []
    for i in idx:
        ret.append(dataset[i])
    return ret

# convert text sample to a string of integers
to_int = np.vectorize(ord)

def format_dataset(dataset):
    return [to_int(list(sample)) for sample in dataset]


def main():
    '''
    parser = argparse.ArgumentParser(description='Program to build and train a neural network.')
    parser.add_argument('--dev_path', default=None, help='Path to development (i.e., testing) data.')
    parser.add_argument('--train_path', default=None, help='Path to the training data directory.')
    parser.add_argument('--max_iters', type=int, default=30, help='The maximum number of EM iterations (default 30)')
    parser.add_argument('--model_out', default=None, help='Filename to save the final model.')
    parser.add_argument('--hidden_states', type=int, default=10,
                        help='The number of hidden states to use. (default 10)')
    args = parser.parse_args()
    
    hmm = HMM(args.hidden_states)
    print('loading datasets...')
    train_dataset = format_dataset(load_subdir(args.train_path))
    test_dataset = format_dataset(load_subdir(args.dev_path))
    print('datasets loaded')
    hmm.train(train_dataset, test_dataset, args.max_iters)
    
    if args.model_out is not None:
        hmm.save_model(args.model_out)
    '''
    
    #dataset = format_dataset(load_sample('C:/Users/Elana/Documents/GitHub/HMM/aclImdbNorm/aclImdbNorm/train/pos/' + '12499_7.txt'))

    print("Loading and parsing data")
    dataset = pick_data(format_dataset(load_subdir('aclImdbNorm/train/pos/')), 100)
    print("Data loaded and parsed")
    
    for i in range(10):
        hmm = HMM(num_states=10)
        hmm.train(dataset, 2)

if __name__ == '__main__':
    main()