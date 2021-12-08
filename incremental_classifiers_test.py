# Evaluator
from skmultiflow.evaluation import EvaluatePrequential

# Classifiers
from skmultiflow.prototype import RobustSoftLearningVectorQuantization
from skmultiflow.anomaly_detection import HalfSpaceTrees
from skmultiflow.trees import HoeffdingTree
from skmultiflow.lazy import KNNClassifier
from skmultiflow.bayes import NaiveBayes
from skmultiflow.meta import AdaptiveRandomForestClassifier

# Data stream generator
from skmultiflow.data import SEAGenerator

# Datasets
from reading_datasets import *

# The SEA generator allows you to generate an infinite data stream
# This particular data stream contains frequent, abrupt concept drift
stream = SEAGenerator(
    random_state=0, balance_classes=True, noise_percentage=0.1
)  # create a stream
stream.prepare_for_use()  # prepare the stream for use

names = [
    "Nearest Neighbors (KNN)",
    "Hoeffding Tree",
    "Random Forest",
    "Anomaly Detection",
    "Naive Bayes",
    "Gradient Boost"
]

classifiers = [
    KNNClassifier(),
    HoeffdingTree(),
    AdaptiveRandomForestClassifier(),
    HalfSpaceTrees(),
    NaiveBayes(),
    RobustSoftLearningVectorQuantization(),
]

nb_iters = 10000  # number of data points to go through

# Setting the show_plot=True option will allow a pop up to appear with
# a real time plot of the classification accuracy.
evaluator = EvaluatePrequential(show_plot=True, max_samples=nb_iters)

# iterate over classifiers
for name, clf in zip(names, classifiers):
    evaluator.evaluate(stream=stream, model=clf)
    print(name)
