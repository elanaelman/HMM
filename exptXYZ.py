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
    num_samples = 1000
    indices = range(1, 11)
    train = False
    do_classify = False
    do_predict = True
    path = 'C:/Users/Elana/Documents/GitHub/HMM/aclImdbNorm/aclImdbNorm/test/'
    #path = 'aclImdbNorm/test/'
    
    if train:
        for i in indices:
            classify.train(i, 20, 100, f'pos_{i}_{max_iter}_{num_samples}', f'neg_{i}_{max_iter}_{num_samples}')
            #TODO: remove already-trained models
    
    if do_classify:
        class_results = []
        for i in indices:
            class_results.append(classify.classify_run(f'models/pos{i}.pickle', f'models/neg{i}.pickle', path, num_samples))
        
        plot("Classification Accuracy vs Hidden States", "Hidden States", "Accuracy", indices, class_results, f"{max_iter} iterations, {num_samples} samples")

    if do_predict:
        pred_results = []
        start_time = 20
        steps = [1, 2, 3, 4, 5]
        
        for i in steps:
            pred_results.append(predict_run("models/pos.pickle", path+'pos', num_samples, start_time, i))
        
        plot("Prediction Accuracy vs Characters Predicted", "Characters Predicted", "Accuracy", steps, pred_results, f"{num_samples} samples")
    

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