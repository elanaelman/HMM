# File: hmm.py
# Purpose:  Starter code for building and training an HMM in CSC 246.

#Code by Elana Elman and Henry Lin
#References: "A Revealing Introduction to Hidden Markov Models" by Mark Stamp; PRML

import argparse  
import os
import numpy as np
from math import inf

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
        #TODO: randomize start state
        self.transitions = np.ones((num_states, num_states))/num_states
        self.emissions = np.ones((num_states, vocab_size))/vocab_size
        self.states = np.arange(num_states)
    
    
    # return the avg loglikelihood for a complete dataset (train OR test) (list of arrays)
    def LL(self, dataset):
        #apply LL_helper to each sample in dataset. Return average.
        return np.average([self.LL_helper(sample) for sample in dataset])

    # return the LL for a single sequence (numpy array)
    def LL_helper(self, sample):
        alpha = self.alpha(sample)
        beta = self.beta(sample)
        alpha, beta, c = self.scale(sample, alpha, beta)
        return -np.sum(np.log(c))
    
    # return the matrix of alphas for a given sequence of observations.
    # alpha[t, i] is the probability of the partial observation sequence up to time t,
    # such that the model is in hidden state i at time t.
    # formula taken from section 4.1 of Stamp's paper.
    def alpha(self, sample):
        #initialize alpha
        a = np.zeros((len(sample), self.num_states))
        # set each alpha at time 0 to 
        # the prior pi(i) times the probability of observing sample[0] in state i.
        a[0] = np.vectorize(lambda i: self.pi[i]*self.emissions[i, sample[0]])(self.states)
        
        #helper function for the interior of the sum term
        def op(t, i, j):
            return a[t, j]*self.transitions[j, i]
        ops = np.vectorize(op)
        #helper function for computing each element
        def a_val(t, i):
            return np.sum(ops(t-1, i, self.states))*self.emissions[i, sample[t]]
        a_vals = np.vectorize(a_val, excluded='t')
        
        for t in range(1, len(sample)):
            #compute alpha at t, i
            a[t] = a_vals(t, self.states)
        
        return a
            
    # return the matrix of betas for a given sequence of observations.
    # beta[t, i] is the probability of observations after time t,
    # given that the model is in state i at time t.
    # formula taken from section 4.2 of Stamp's paper.
    def beta(self, sample):
        #initialize beta
        b = np.ones((len(sample), self.num_states))
        
        #helper function for the interior of the sum term
        def op(t, i, j):
            return b[t, j]*self.transitions[i, j]*self.emissions[j, sample[t]]
        ops = np.vectorize(op)
        #helper function for computing the sum term
        def sum_ops(t, i):
            return np.sum(ops(t, i, self.states))
        
        #todo: vectorize
        for t in range(len(sample)-2, -1, -1): #go backwards through indices
            for i in self.states:
                #compute beta at t, i.
                b[t, i] = sum_ops(t+1, i)
            
        return b
    
    def scale(self, sample, alpha, beta):
        c = np.zeros((len(sample)))
        a_tilde = np.zeros(alpha.shape)
        a_tilde[0] = alpha[0]
        
        c[0] = 1/np.sum(alpha[0])
        alpha[0] = c[0]*a_tilde[0]
        for t in range(1, len(sample)):
            for i in range(1, self.num_states):
                product = lambda j: alpha[t-1, j]*self.transitions[j, i]*self.emissions[i, sample[t]]
                a_tilde[t, i] = np.sum(np.vectorize(product)(range(self.num_states)))
            c[t] = 1/np.sum(a_tilde[t])
            alpha[t] = c[t]*a_tilde[t]
            beta[t] = c[t]*beta[t]
        return alpha, beta, c
    
    # return the matrix of gammas for a given sequence of observations.
    # gamma[t, i] is the probability of being in state i at time t, given the observations.
    # formula taken from section 4.2 of Stamp's paper.
    def gamma(self, sample, alpha = None, beta = None):
        #first calculate alpha and beta, if needed.
        if alpha is None:
            alpha = self.alpha(sample)
        if beta is None:
            beta = self.beta(sample)
            
        #Calculate the total probability of the observations:
        p_o = np.sum(alpha[len(sample)-1])
        
        #helper function defining gamma for each term
        def g(t, i):
            return alpha[t, i]*beta[t, i]/p_o
        gs = np.vectorize(g)
        
        #construct matrix
        return gs(np.transpose([range(len(sample))]), [self.states])
    
    # return the most likely state at time t based on a sequence of observations
    # formula from section 4.2 of Stamp's paper.
    def guess_state(self, sample, t, gamma = None):
        #construct gamma if necessary
        if gamma == None:
            gamma = self.gamma(sample, t)
        return np.argmax(gamma)
        
    # Return the matrix of gammas for a given sequence of observations.
    # gamma[t, i, j] is the probability of being in state i at time t and then 
    # transitioning to state j at time t+1, given a sequence of observations.
    # Formula from section 4.3 of Stamp's paper.
    def di_gamma(self, sample, alpha, beta):
        d_g = np.zeros((len(sample), self.num_states, self.num_states))
        
        p_o = np.sum(alpha[len(sample)-1])
        def d_g(t, i, j):
            return alpha[t, i]*self.transitions[i, j]*self.emissions[j, sample[t+1]]*beta[t+1, j]/p_o
        d_gs = np.vectorize(d_g)
        
        return d_gs(np.array([[range(len(sample)-1)]]).transpose(2, 0, 1), 
                    np.transpose([[self.states]], (0, 2, 1)), 
                    [[self.states]])

    # apply a single step of the em algorithm to the model on all the training data,
    # which is most likely a python list of numpy matrices (one per sample).
    # Note: you may find it helpful to write helper methods for the e-step and m-step,
    def em_step(self, dataset):
        #todo: for each sample, or something
        sample = dataset[0]
        
        T = len(sample)
        alpha = self.alpha(sample)
        beta = self.beta(sample)
        alpha, beta, c = self.scale(sample, alpha, beta)
        di_gamma = self.di_gamma(sample, alpha, beta)
        gamma = self.gamma(sample, alpha, beta)
        
        ts = range(T-1)
        
        #update pi:
        self.pi = gamma[0]
            
        #update A:
        def t(i, j):
            numerator = np.sum(di_gamma[ts, i, j])
            denominator = np.sum(gamma[ts, j])
            return numerator/denominator
        t_vectorized = np.vectorize(t)
        
        #update B:
        def e(j, k):
            indices = np.nonzero(sample == k) #get list of times where the observation is k
            numerator = np.sum(gamma[indices, j])
            denominator = np.sum(gamma[ts, j])
            return numerator/denominator
        e_vectorized = np.vectorize(e)
        
        #update model
        self.transitions = t_vectorized(np.transpose([self.states]), [self.states])
        self.emissions = e_vectorized(np.transpose([self.states]), [range(self.vocab_size)])
        
        #return updated probability
        return self.LL(dataset)
    
    #todo: full dataset
    def train(self, sample, maxIters):
        dataset = [sample]
        
        oldLL = -inf
        for i in range(maxIters):
            self.em_step(dataset)
            newLL = self.LL(dataset)
            print(newLL)
            if newLL > oldLL:
                oldLL = newLL
            else:
                break
            

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

#load a single review
def load_sample(file):
    o = open(file)
    f = o.read()
    o.close()
    return f
        
#convert text sample to a string of integers
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
    #filepath for elana:
    file = "C:/Users/Elana/Documents/GitHub/HMM/aclImdbNorm/aclImdbNorm/train/pos/10551_7.txt"
    # file = "aclImdbNorm/train/pos/10551_7.txt"
    hmm = HMM(num_states=2)
    sample = format_sample(load_sample(file))
    hmm.train(sample, 5)
    #main()
