# File: hmm.py
# Purpose:  Starter code for building and training an HMM in CSC 246.

#references: "A Revealing Introduction to Hidden Markov Models" by Mark Stamp; PRML

import argparse  
import os
import numpy as np

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
    def __init__(self, num_states = 10, vocab_size = 255):
        self.num_states = num_states
        self.vocab_size = vocab_size
        self.pi = np.ones(num_states)/num_states
        #randomize start state?
        self.transitions = np.ones((num_states, num_states))/num_states
        self.emissions = np.ones((num_states, vocab_size))/vocab_size
        self.states = np.arange(num_states)
    
    
    # return the avg loglikelihood for a complete dataset (train OR test) (list of arrays)
    #TODO: move vectorized fn to class field
    def LL(self, dataset):
        return np.avg(np.vectorize(self.p)(dataset))

    # return the LL for a single sequence (numpy array)
    def LL_helper(self, sample):
        return self.p(sample)
    
    def p(self, sample):
        return np.sum(self.alpha(sample))
    
    def alpha(self, sample):
        a = np.zeros((len(sample), self.num_states))
        a[0] = np.vectorize(lambda i: self.pi[i]*self.emissions[i, sample[0]])(self.states)
        
        
        def op(t, i, j):
            return a[t, j]*self.transitions[j, i]
        ops = np.vectorize(op)
        def sum_ops(t, i):
            return np.sum(ops(t, i, self.states))
        
        for t in range(1, len(sample)):
            for i in self.states:
                a[t, i] = sum_ops(t-1, i)*self.emissions[i, sample[t]]
        
        return a
            
    
    def beta(self, sample):
        b = np.ones((len(sample), self.num_states))
        
        def op(t, i, j):
            return b[t, j]*self.transitions[i, j]*self.emissions[j, sample[t]]
        ops = np.vectorize(op)
        def sum_ops(t, i):
            return np.sum(ops(t, i, self.states))
        
        for t in range(len(sample)-2, -1, -1): #go backwards through indices
            for i in self.states:
                b[t, i] = sum_ops(t+1, i)
            
        return b
    
    def guess_state(self, sample, t):
        a = self.alpha(sample)
        b = self.beta(sample)
        p_o = np.sum(a[len(sample)-1])
        gamma = np.vectorize(lambda t, i: a[t, i]*b[t, i]/p_o)(t, self.states)
        return np.argmax(gamma)
        
            

    # apply a single step of the em algorithm to the model on all the training data,
    # which is most likely a python list of numpy matrices (one per sample).
    # Note: you may find it helpful to write helper methods for the e-step and m-step,
    def em_step(self, dataset):
        pass

    # Return a "completed" sample by additing additional steps based on model probability.
    def complete_sequence(self, sample, steps):
        pass


    # Save the complete model to a file (most likely using np.save and pickles)
    def save_model(self, filename):
        pass

# Load a complete model from a file and return an HMM object (most likely using np.load and pickles)
def load_hmm(filename):
    pass

# Load all the files in a subdirectory and return a giant list.
def load_subdir(path):
    data = []
    for filename in os.listdir(path):
        with open(os.path.join(path, filename)) as fh:
            data.append(fh.read())
    return data

def load_sample(file):
    o = open(file)
    f = o.read()
    o.close()
    return f
        

to_int = np.vectorize(ord)
def format_sample(sample):
    return to_int(list(sample))

def main():
    parser = argparse.ArgumentParser(description='Program to build and train a neural network.')
    parser.add_argument('--dev_path', default=None, help='Path to development (i.e., testing) data.')
    parser.add_argument('--train_path', default=None, help='Path to the training data directory.')
    parser.add_argument('--max_iters', type=int, default=30, help='The maximum number of EM iterations (default 30)')
    parser.add_argument('--model_out', default=None, help='Filename to save the final model.')
    parser.add_argument('--hidden_states', type=int, default=10, help='The number of hidden states to use. (default 10)')
    args = parser.parse_args()
    
    hmm = HMM()
    hmm.p(0)

    # OVERALL PROJECT ALGORITHM:
    # 1. load training and testing data into memory
    #
    # 2. build vocabulary using training data ONLY
    #
    # 3. instantiate an HMM with given number of states -- initial parameters can
    #    be random or uniform for transitions and inital state distributions,
    #    initial emission parameters could bea uniform OR based on vocabulary
    #    frequency (you'll have to count the words/characters as they occur in
    #    the training data.)
    #
    # 4. output initial loglikelihood on training data and on testing data
    #
    # 5+. use EM to train the HMM on the training data,
    #     output loglikelihood on train and test after each iteration
    #     if it converges early, stop the loop and print a message

if __name__ == '__main__':
    file = "C:/Users/eelman2/Downloads/aclImdbNorm/aclImdbNorm/train/pos/10551_7.txt"
    hmm = HMM(num_states=2)
    sample = format_sample(load_sample(file))
    print(hmm.guess_state(sample, 1))
    #main()
