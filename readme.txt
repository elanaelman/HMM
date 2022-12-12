CSC 246 Project 3: HMM
Henry Lin and Elana Elman

Our program has three parts.
    1. hmm.py
        This program builds, trains, and saves a HMM based on training data.
        The command to run it is:
            python3 hmm.py --dev_path DEV_PATH --train_path TRAIN_PATH --max_iters ITERS --model_out PATH_OUT --hidden_states STATES
        By default, it runs with 10 hidden states and a maximum of 100 iterations.
        
    2. classify.py
        This program uses pre-existing models trained respectively on positive and negative data 
        to predict whether a sample is positive or negative. 
        The command to run it is:
            python3 classify.py --pos_hmm POS_PATH --neg_hmm NEG_PATH --datapath TEST_DATA
            
    3. predict.py
        This program evaluates the predictive ability of a pre-existing model.
        It looks at the first t characters of each sample in the testing dataset, and
        calculates the proportion of samples for which the model correctly predicts the next character.
        The command to run it is:
            python3 predict.py --hmm HMM_PATH --test_data TEST_PATH --t TIME_TO_GUESS