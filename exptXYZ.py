# Experiments

from hmm import *
from predict import *
import classify
import matplotlib.pyplot as plt


def main():
    #train(10, 20, 10)
    #predict_run('pos.pickle', 'aclImdbNorm/test/pos/', 100)
    #classify_run('pos.pickle', 'neg.pickle','aclImdbNorm/test', 100)

    max_iter = 20
    num_samples = 100
    indices = [1, 2, 5, 7, 10]
    train = False
    #path = 'C:/Users/Elana/Documents/GitHub/HMM/aclImdbNorm/aclImdbNorm/test/'
    path = 'aclImdbNorm/test/'
    
    if train:
        for i in indices:
            classify.train(i, 20, 100, f'pos_{i}_{max_iter}_{num_samples}', f'neg_{i}_{max_iter}_{num_samples}')
            #TODO: remove already-trained models
    
    results = []
    for i in indices:
        results.append(classify.classify_run(f'pos_{i}_{max_iter}_{num_samples}', f'neg_{i}_{max_iter}_{num_samples}', path, num_samples))
    
    plot("Classification Accuracy vs Hidden States", "Hidden States", "Accuracy", indices, results, f"{max_iter} iterations, {num_samples} samples")


def plot(title, xlabel, ylabel, x_list, y_list, meta):
    plt.clf()
    plt.title(f"{title} ({meta})")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x_list)
    plt.plot(x_list, y_list, label="Accuracy on Testing Data")
    plt.legend()
    plt.savefig(f"{title} ({meta}).png")


if __name__ == '__main__':
    main()