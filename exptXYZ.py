# Experiments

from hmm import *
from predict import *
from classify import *
import os


def main():
    #train(10, 20, 10)
    predict_run('pos.pickle', 'aclImdbNorm/test/pos/', 100)
    #classify_run('pos.pickle', 'neg.pickle','aclImdbNorm/test', 100)







if __name__ == '__main__':
    main()