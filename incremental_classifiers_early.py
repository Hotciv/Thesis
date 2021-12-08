from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.trees import HoeffdingTree
from skmultiflow.data import SEAGenerator
import matplotlib.pyplot as plt

# The SEA generator allows you to generate an infinite data stream
# This particular data stream contains frequent, abrupt concept drift
stream = SEAGenerator(random_state=0, balance_classes=True, noise_percentage=0.1)  # create a stream
stream.prepare_for_use()  # prepare the stream for use

X, Y = stream.next_sample()
# print(X, Y)
# input()

# Hoeffding trees³ are built using the Very Fast Decision Tree Learner (VFDT),
# an anytime system that builds decision trees using constant memory and
# constant time per example. Introduced in 2000 by Pedro Domingos and Geoff Hulten,
# it makes use of a well known statistical result, the Hoeffding bound,
# in order to guarantee that its output is asymptotically identical to
# that of a traditional learner.
tree = HoeffdingTree()

nb_iters = 1000  # number of data points to go through

# correctness_dist = []
# for i in range(nb_iters):
#     X, Y = stream.next_sample()  # get the next sample
#     prediction = tree.predict(X)  # predict Y using the tree

#     if Y == prediction:  # check the prediction
#         correctness_dist.append(1)
#     else:
#         correctness_dist.append(0)

#     tree.partial_fit(X, Y)


# time = [i for i in range(1, nb_iters)]
# accuracy = [sum(correctness_dist[:i])/len(correctness_dist[:i]) for i in range(1, nb_iters)]
# print(accuracy[-1])
# plt.plot(time, accuracy)
# plt.show()

# Setting the show_plot=True option will allow a pop up to appear with
# a real time plot of the classification accuracy.
evaluator=EvaluatePrequential(show_plot=True,max_samples=nb_iters)

evaluator.evaluate(stream=stream, model=tree)