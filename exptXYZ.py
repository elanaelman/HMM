# Experiments

from hmm import *
from predict import *
import classify
import matplotlib.pyplot as plt


def main():

    max_iter = 20
    num_samples = 1000
    indices = [1, 2, 3, 5, 7, 10]
    path = 'aclImdbNorm/test/'
    
    if train:
        for i in indices:
            classify.train(i, 20, 100, f'models/pos{i}.pickle', f'models/neg{i}.pickle')
    
    if do_classify:
        class_results = []
        for i in indices:
            class_results.append(classify.classify_run(f'models/pos{i}.pickle', f'models/neg{i}.pickle', path, num_samples))
        
        plot("Classification Accuracy vs Hidden States", "Hidden States", "Accuracy", indices, class_results, f"{max_iter} iterations, {num_samples} samples")

    if do_predict:
        pred_results = []
        start_time = 20
        
        #for i in steps:
        #    pred_results.append(predict_run("models/pos.pickle", path+'pos', num_samples, start_time, i))
        vals = predict_run("models/pos7.pickle", path+'pos', num_samples, start_time)
        
        hist("Longest Correct Predictions", "Characters Predicted", vals, f"{num_samples} samples")
    

def plot(title, xlabel, ylabel, x_list, y_list, meta):
    plt.clf()
    plt.title(f"{title} ({meta})")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x_list)
    plt.plot(x_list, y_list, label="Accuracy on Testing Data")
    plt.legend()
    plt.savefig(f"{title} ({meta}).png")
    
def hist(title, xlabel, y_list, meta):
    plt.clf()
    plt.title(f"{title} ({meta})")
    plt.xlabel(xlabel)
    counts, edges, bars = plt.hist(y_list, bins=[0, 1, 2, 3, 4])
    plt.xticks([0, 1, 2, 3, 4])
    plt.bar_label(bars)
    plt.savefig(f"{title} ({meta}).png")


if __name__ == '__main__':
    main()