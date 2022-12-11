# File: classify.py
# Purpose:  Starter code for the main experiment for CSC 246 P3 F22.

import argparse
from hmm import *


def classify_run(pos_hmm=None, neg_hmm=None, path=None, sample_size=10):

    if __name__ == '__main__':
        args = classify_args()
        if args.pos_hmm is None and args.neg_hmm is None:
            train()
            args.pos_hmm = "pos.pickle"
            args.neg_hmm = "neg.pickle"
        else:
            # Load HMMs
            pos_hmm = load_hmm(args.pos_hmm)
            neg_hmm = load_hmm(args.neg_hmm)
        datapath = args.datapath
    else:
        pos_hmm = load_hmm(pos_hmm)
        neg_hmm = load_hmm(neg_hmm)
        datapath = path

    correct = 0
    total = 0

    # test samples from positive datapath
    samples = pick_data(format_dataset(load_subdir(os.path.join(datapath, 'pos'))), sample_size)
    for sample in samples:
        if pos_hmm.LL_helper(sample) > neg_hmm.LL_helper(sample):
            correct += 1
        total += 1

    # test samples from negative datapath
    samples = pick_data(format_dataset(load_subdir(os.path.join(datapath, 'neg'))), sample_size)
    for sample in samples:
        if pos_hmm.LL_helper(sample) < neg_hmm.LL_helper(sample):
            correct += 1
        total += 1

    # report accuracy  (no need for F1 on balanced data)
    print("%d/%d correct; accuracy %f" % (correct, total, correct / total))


def train(states = 10, max_iters=20, sample_size=10):
    # pos_dataset = format_dataset(load_subdir('C:/Users/Elana/Documents/GitHub/HMM/aclImdbNorm/aclImdbNorm/train/pos/'))
    # neg_dataset = format_dataset(load_subdir('C:/Users/Elana/Documents/GitHub/HMM/aclImdbNorm/aclImdbNorm/train/neg/'))
    pos_dataset = pick_data(format_dataset(load_subdir('aclImdbNorm/train/pos/')), sample_size)
    neg_dataset = pick_data(format_dataset(load_subdir('aclImdbNorm/train/neg/')), sample_size)
    pos_hmm = HMM(num_states=states)
    neg_hmm = HMM(num_states=states)
    pos_hmm.train(pos_dataset, max_iters)
    neg_hmm.train(neg_dataset, max_iters)
    pos_hmm.save_model('pos.pickle')
    neg_hmm.save_model('neg.pickle')


def classify_args():
    parser = argparse.ArgumentParser(description='Program to build and train a neural network.')
    parser.add_argument('--pos_hmm', default=None, help='Path to the positive class hmm.')
    parser.add_argument('--neg_hmm', default=None, help='Path to the negative class hmm.')
    parser.add_argument('--datapath', default=None, help='Path to the test data.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    classify_run()