# Victor Barboza thesis python program

This is the program used in the Master Thesis titled "Analysis of Selection Bias in Online Adversarial Aware Machine Learning Systems" by Victor Barboza, 2022.

Four experiments were conducted:

1. Train using classifiers from sklearn
2. Train using SGDOneClassSVM on an online, sliding window setting
3. Train using AAOSVM without 200 samples
4. Train using AAOSVM without 200 selected samples according to support vectors from:
    1. LinSVM from experiment 1
    2. AAOSVM from experiment 3

To run the experiments yourself, run experiments.py using the command[*]

    python run experiments.py

To begin, select which datasets do you want to use. Pressing `enter` without entering any selection will use all five datasets.

Next, you must select which of the four experiments are you going to be running.

Next, you must select between running the experiment 10x or 1x. Pressing `enter` without entering any selection will run the experiment 10x.

Next, you must decide if the datasets are going to be used as is or if they are going to be normalized using MinMax normalization.

Finally, you must select if the experiment is being loaded or runned for the first time.
* If the experiments are loaded, the csv file with the perfomance results can be rebuild from the saved models.
* The bias metrics are only calculated with the loaded models.

To calculate all the metrics, the program must run twice. First to train the model(s), calculate its(their) performance, and save both the trained model and the result of the performance. Second to calculate and save the bias metrics.

[*]: The datasets must be kept in the same arrangement as they are and you must have a folder under the same name as listed on `aux_functions.py` in the variable `results_dir`.