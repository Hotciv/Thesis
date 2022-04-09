##########################################
#############--- imports ---##############
##########################################

from reading_datasets import *
from aux_functions import *
from csv import writer
from time import time
from glob import glob

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import SGDOneClassSVM

from AAOSVM import *

from adversarial_samples import generating_adversarial_samples, generating_labels, cost

# # for helping the experiment 0?
# from pyspark.sql import SparkSession # instantiate spark session
# from skdist.distribute.search import DistGridSearchCV
# # /for helping the experiment 0?

##########################################
##########--- general settings---#########
##########################################

np.random.seed(0)

print("Which datasets do you wish to use? [all]")
print("0 -> ds1")
print("1 -> ds2")
print("2 -> ds3")
print("3 -> ds4")
print("4 -> ds5")
print("\tUsage examples:")
print("\t\t'123' will use ds2, ds3, and ds4")
print("\t\t'13' will use ds2 and ds4")
print("\t\tpressing 'enter' without typing anything will use all")
df_numbers = list(input() or "01234")
valid = False
while not valid:
    valid = True
    for df in df_numbers:
        if df != "0" and df != "1" and df != "2" and df != "3" and df != "4":
            print("Please select a valid input!\n")
            print("Which datasets do you wish to use? [all]")
            print("0 -> ds1")
            print("1 -> ds2")
            print("2 -> ds3")
            print("3 -> ds4")
            print("4 -> ds5")
            df_numbers = list(input() or "01234")
            valid = False


df_numbers = [int(x) for x in df_numbers]
datasets, sel_names = to_dataset(df_numbers)

# assembly of the 'name'
name = ""
for s in sel_names:
    name = name + s + ", "
name = name[:-2]

# input -> experiment number
n = int(input("What is the experiment? [1-4]\n"))
while n < 1 and n > 4:
    n = int(input("What is the experiment? [1-4]\n"))


kn = input("Repeat experiments 10x? ([y]/n)\n") or "y"
while kn != "y" and kn != "n":
    kn = input("Repeat experiments 10x? ([y]/n)\n") or "y"
if kn == "y":
    kn = 10
else:
    kn = 1

# input -> normalize datasets?
normalization = input("Use normalized dataset? (y/n)\n")
while normalization != "y" and normalization != "n":
    normalization = input("Use normalized dataset? (y/n)\n")

if normalization == "y":
    # iterate over datasets and normalize each one of them
    for ds_cnt, ds in enumerate(datasets):
        X, y = ds
        datasets[ds_cnt] = (dataset_normalization([X])[0], y)

# input -> create a new csv file?
load = input("Loading trained models? (y/n)\n")
while load != "y" and load != "n":
    load = input("Loading trained models? (y/n)\n")

rerun = False
if load == "y":
    # input -> rebuild csv file?
    rebuild = input("Rebuild csv file from trained models? (y/n)\n")
    while rebuild != "y" and rebuild != "n":
        rebuild = input("Rebuild csv file from trained models? (y/n)\n")
    if rebuild == "y":
        rerun = True

    # input -> choice of support vectors
    if n == 4 and not rerun:
        op = input("Which support vectors do you wish to use? ((b)atch/(a)aosvm)\n")
        while op != "a" and op != "b":
            op = input("Which support vectors do you wish to use? ((b)atch/(a)aosvm)\n")
        # TODO: remove this
        # normalization = "t"


def filename_build(n: int, selection="x random200"):
    """
    Function to create an automated filename for the experiments
    """
    results = glob(results_dir + "/*/")
    if normalization == "y":
        norm = ", normalized"
        r = 1
    else:
        norm = ""
        r = 0

    if rerun:
        rr = ", rerun"
    else:
        rr = ""

    if n == 1:
        fn = "standard_classifiers_results "
    elif n == 2:
        fn = "OSVM, "
    elif n == 3:
        fn = "AAOSVM, "
    elif n == 4:
        fn = "AAOSVM_ch_scores, "

    return results[r] + fn + name + " " + str(kn) + selection + norm + rr + ".csv"


##############################################
################--- TODO: ---#################
# --- experiment 0 - getting the best shot ---#
##############################################

# experiment 0.1 - normalization
# experiment 0.2 - tunning

# # For DistGridSearchCV to work
# spark = (
#     SparkSession
#     .builder
#     .getOrCreate()
#     )
# sc = spark.sparkContext

# # somewhere in the training bits
# param_grid = {}
# ACCs = DistGridSearchCV(
#             clf, param_grid,
#             sc=sc, scoring='accuracy',
#             verbose=True
#             )

# experiment 0.3 - tunning on normalization

##########################################
# --- experiment 1 - standard, batch ---###
##########################################


def experiment_cv(
    n: int, wrt: writer, X, y, k=11, just_SVMs=False, selection="10x random200"
):
    """
    The core of experiments cross validation and saves results in\
    csv files with filenames automatically generated.

    Parameters:
        n (int): experiment number.
        wrt (writer): csv writer to save the results.
        k (int): iteration number; 11 is for when there is just one execution.
        just_SVMs
            (bool): as False, uses all classifiers available and\
                as True, uses just SVMs.
            (list): list of indexes of classifiers to be used.
        selection (["10x random200", "support_vectors"]): selection of\
            random 200 phishing samples or\
            support vectors + close by samples (KNN)
    """
    if selection == "10x random200":
        # removing 200 random phishing samples
        X_train, _, y_train, _, _ = dataset_split(X, y, 200, 1, k)

    # settings
    upd = False
    if n == 1:
        names = [
            "Nearest Neighbors (KNN)",
            "Linear SVM",
            "RBF SVM",
            "Decision Tree",
            "Random Forest",
            "Naive Bayes",
            "Gradient Boost",
        ]

        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=100, max_iter=10000, random_state=42),
            SVC(gamma=2, C=100, max_iter=10000, random_state=42),
            DecisionTreeClassifier(max_depth=5, random_state=42),
            RandomForestClassifier(
                max_depth=5, n_estimators=10, max_features=1, random_state=42
            ),
            GaussianNB(),
            GradientBoostingClassifier(random_state=42),
        ]

        clf_type = "std"

        if just_SVMs == True:
            names = names[1:3]
            classifiers = classifiers[1:3]

    elif n == 2:
        names = ["inc"]
        classifiers = [SGDOneClassSVM(random_state=42)]
        clf_type = "OSVM"

    elif n == 3 or n == 4:
        names = ["AAOSVM"]

        # AAOSVM intitalization
        # Set model parameters and initial values
        C = 100.0
        m = 2
        initial_alphas = np.zeros(m)
        initial_b = 0.0
        initial_w = np.zeros(len(ds[0][0]))

        # Slidding window
        Sx = X_train[0]
        Sy = y_train[0]
        Sx = np.append([Sx], [X_train[1]], 0)
        Sy = np.append([Sy], [y_train[1]], 0)

        # Initialize model
        if n == 4:
            upd = True
        model = AAOSVM(Sx, Sy, C, initial_alphas, initial_b, np.zeros(m), update=upd)

        # Initialize error cache
        initial_error = model.predict(model.X) - model.y
        model.errors = initial_error

        # Initialize weights
        model.w = initial_w
        # /AAOSVM intitalization

        classifiers = [model]
        clf_type = "AAOSVM"
    # /settings

    # iterate over classifiers
    for clf_name, clf in zip(names, classifiers):

        print(clf_name)

        # getting initial parameters to reset classifier in the cross validation
        if n == 1 or n == 2:
            reset = clf.get_params()
        else:
            reset = None

        # timed cross validation
        start_time = time()
        ACCs, TPRs, F1s, loss = cross_validate(
            clf,
            X_train,
            y_train,
            clf_type,
            clf_name,
            random_state=k,
            aux="_" + sel_names[ds_cnt],
            reset=reset,
            normalization=normalization,
            update=upd,
        )
        finish = time() - start_time

        print("finished", finish)

        # saving
        # header  # to peek at definition
        wrt.writerow(
            [
                sel_names[ds_cnt],
                k,
                clf_name,
                ACCs.mean(),
                ACCs.std(),
                TPRs.mean(),
                TPRs.std(),
                F1s.mean(),
                F1s.std(),
                finish,
                loss,
            ]
        )
        print("wrote")


def experiment_load(
    n: int, wrt=None, k=11, all_models=False, just_SVMs=False, selection="10x random200"
):
    """
    Load experiments from cross validation and returns model, the selected 200 samples, 
    and the test/train from the original cross validation split.

    Parameters:
        n (int): experiment number.
        wrt (writer): csv writer to save the results of the rerun.
        k (int): iteration number; 11 is for when there is just one execution.
        just_SVMs
            (bool): as False, uses all classifiers available and
                as True, uses just SVMs.
            (list): list of indexes of classifiers to be used.
        selection (["10x random200", "support_vectors"]): selection of
            random 200 phishing samples or
            support vectors + close by samples (KNN)

    Yields:
        clf 
            (classifier): trained classifier from a cross validation.
            (list): all trained classifiers from a cross validation.
        the_200 (np.ndarray): the selected 200 samples.
        X_test (np.ndarray): split of the dataset to test.
        y_test (np.array): split of the labels of the dataset to test.
        selected (list/np.array): indexes of the test split.
    """
    results = glob(results_dir + "/*/")
    if normalization == "y":
        r = 1
    elif normalization == "n":
        r = 0
    else:
        r = 2

    if selection == "10x random200":
        # removing 200 random phishing samples
        X_train, X_test, y_train, y_test, selected = dataset_split(X, y, 200, 1, k)
        the_200 = X_test

    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    # settings
    if n == 1:
        names = [
            "Nearest Neighbors (KNN)",
            "Linear SVM",
            "RBF SVM",
            "Decision Tree",
            "Random Forest",
            "Naive Bayes",
            "Gradient Boost",
        ]

        clf_type = "std"

        if just_SVMs == True:
            names = names[1:3]
            classifiers = classifiers[1:3]

    elif n == 2:
        names = ["inc"]
        clf_type = "OSVM"

    elif n == 3 or n == 4:
        names = ["AAOSVM"]
        clf_type = "AAOSVM"
    # /settings

    # iterate over classifiers
    for clf_name in names:
        # loading the first model from a pkl file
        f = open(
            results[r]
            + clf_name
            + "_"
            + sel_names[ds_cnt]
            + "_{}_{}.pkl".format(clf_type, k),
            "rb",
        )

        if not all_models:
            # loads first model from a cross validation
            clf = pickle.load(f)

            # gets the test/train from the original cross validation split
            train_index, test_index = kf.split(X_train, y_train).__next__()
            _, X_hold = X_train[train_index], X_train[test_index]
            _, y_hold = y_train[train_index], y_train[test_index]
        else:
            clf = []
            splits = kf.get_n_splits()
            ACCs = np.zeros(splits)
            TPRs = np.zeros(splits)
            F1s = np.zeros(splits)
            loss = np.zeros(splits)
            predicted_time = np.zeros(splits)
            j = 0
            start_time = time()
            for train_index, test_index in kf.split(X_train, y_train):
                _, X_hold = X_train[train_index], X_train[test_index]
                _, y_hold = y_train[train_index], y_train[test_index]

                # loads all models from a cross validation, one by one
                clf.append(pickle.load(f))
                print(clf_name, time() - start_time, j, "loaded")
                start_time = time()

                y_pred = clf[j].predict(X_hold)
                predicted_time[j] = time() - start_time
                print(clf_name, predicted_time[j], j, "predicted")
                start_time = time()
                neg = y_pred == -1
                pos = y_pred == 1
                bin = neg | pos

                ACCs[j] = accuracy_score(y_hold[bin], y_pred[bin])
                TPRs[j] = recall_score(y_hold[bin], y_pred[bin])
                F1s[j] = f1_score(y_hold[bin], y_pred[bin])
                loss[j] = len(y_pred[~bin])

                j += 1

            # header  # to peek at def
            wrt.writerow(
                [
                    sel_names[ds_cnt],
                    k,
                    clf_name,
                    ACCs.mean(),
                    ACCs.std(),
                    TPRs.mean(),
                    TPRs.std(),
                    F1s.mean(),
                    F1s.std(),
                    predicted_time.mean(),
                    loss,
                ],
            )

        yield clf, the_200, X_hold, y_hold, selected, clf_name


def send_noise(
    clf: Union[object, list],
    clf_name: str,
    the_200: np.ndarray,
    X: np.ndarray,
    y: np.array,
    k: int,
    ds_cnt: int,
):
    """
    Gets 200 random samples or 200 samples based of support vectors\
        and use them to compute bias and other metrics.

    Parameters:
        clf (object): classifier to go against the AST.
        clf_name (str): name of the classifier.
        the_200 (np.ndarray): 
        X (np.ndarray): dataset to be used.
        y (np.array): labels of the dataset to be used.
        k (int): iteration number; used as initial random\
            state for auditatorial purposes.
        ds_cnt (int): number of current dataset, for filename\
            build purposes.
    """
    results = glob(results_dir + "/*/")
    if normalization == "y":
        r = 1
    elif normalization == "n":
        r = 0
    else:
        r = 2

    if n == 4:
        if op == "a":
            fn = glob(results[r] + "indexes - _" + sel_names[ds_cnt] + "_0_0*.pkl")[0]
            support_indexes = get_indexes(fn)
        elif op == "b":
            fn = glob(results[r] + "Linear SVM_" + sel_names[ds_cnt] + "_std_0.pkl")[0]
            support_indexes = get_indexes(fn)

        y_ = y[support_indexes]

        _, the_200, _, _, _ = dataset_split(X, y, 200, support_indexes[y_ == 1])

    header = [
        "Number of manipulated features",
        "Criteria for generating labels",
        "Class Imbalance",
        "DPPTL mean [0]",
        "DPPTL mean [1]",
        "DPPTL std [0]",
        "DPPTL std [1]",
        "Distance from original sample (average)",
        "Distance from original sample (std)",
        "Percentage of FP",
    ]

    if n == 3:
        fn_partial = "AAOSVM, "
    elif n == 4:
        fn_partial = "AAOSVM_ch_scores "
        if op == "a":
            fn_partial = fn_partial + "with self sup_vect, "
        elif op == "b":
            fn_partial = fn_partial + "with batch sup_vect, "

    fn = (
        results[r]
        # + clf_name
        + fn_partial
        + " metrics - "
        + sel_names[ds_cnt]
        + "_{}_{}.csv".format(k, datetime.now().strftime("%Y%m%d %H%M%S"))
    )

    # saving the results on a csv file
    g = open(fn, "w", newline="")
    wrt_mtrc = writer(g)
    wrt_mtrc.writerow(header)

    dist = []

    pred = clf.predict(X)
    # feature selection and generating adversarial samples
    for f in range(3):
        for c in range(3):
            if f == 0:
                gen_labels = generating_labels(the_200, c)
                y_pred = clf.predict(the_200)

                # calculating bias metrics
                y_aux = np.append(y, gen_labels)
                ci = class_imbalance(y_aux)
                dp = np.nan

                # recording distances
                dist = np.nan

                # number of instances that gets classified as regular
                bypass = len(y_pred[y_pred == -1]) / len(y_pred)

                wrt_mtrc.writerow(
                    (f, c, ci, dp, np.nan, np.nan, np.nan, dist, np.nan, bypass)
                )
            else:
                bypass = 0
                dist = []
                dp = set()
                for x in the_200:
                    sel_features = feature_selection(x, f, k)
                    gen_samples, gen_labels = generating_adversarial_samples(
                        x, sel_features, X, y, c
                    )
                    y_pred = clf.predict(gen_samples)

                    # calculating bias metrics
                    y_aux = np.append(y, gen_labels)
                    X_aux = np.append(X, gen_samples, axis=0)
                    pred_aux = np.append(pred, y_pred)
                    ci = class_imbalance(y_aux)
                    d_aux = set()
                    for feat in sel_features:
                        d_aux.add(DPPTL(feat, gen_samples[:, feat][0], X_aux, pred_aux))
                    dp.add(tuple(d_aux))

                    # does any instance pass as legitimate?
                    if np.any(y_pred == -1):
                        bypass += 1

                        # recording distances
                        d_aux = []
                        for gen_s in gen_samples:
                            cst = cost(x, gen_s)
                            if cst != 0:
                                d_aux.append(cst)
                        dist.append(min(d_aux))

                    else:
                        dist.append(np.nan)

                dist = np.array(dist)
                dist = dist[~np.isnan(dist)]
                dp = np.array(list(dp))
                if len(dp) == 1:
                    wrt_mtrc.writerow(
                        (
                            f,
                            c,
                            ci,
                            dp[0],
                            np.nan,
                            np.nan,
                            np.nan,
                            dist.mean(),
                            dist.std(),
                            bypass / 200,
                        )
                    )
                elif f == 1:
                    wrt_mtrc.writerow(
                        (
                            f,
                            c,
                            ci,
                            dp.mean(axis=0)[0],
                            np.nan,
                            dp.std(axis=0)[0],
                            np.nan,
                            dist.mean(),
                            dist.std(),
                            bypass / 200,
                        )
                    )
                else:
                    wrt_mtrc.writerow(
                        (
                            f,
                            c,
                            ci,
                            dp.mean(axis=0)[0],
                            dp.mean(axis=0)[1],
                            dp.std(axis=0)[0],
                            dp.std(axis=0)[1],
                            dist.mean(),
                            dist.std(),
                            bypass / 200,
                        )
                    )
                print(f, c, gen_labels.shape)
    g.close


# Creating a csv writter to save the results
if load == "n" or (load == "y" and rerun):
    header = [
        "Dataset",
        "Run",
        "Classifier",
        "ACC",
        "ACC std",
        "TPR",
        "TPR std",
        "F1",
        "F1 std",
        "Time to execute",
        "Loss",
    ]
    # saving the results on a csv file
    f = open(filename_build(n), "w", newline="")
    wrt = writer(f)
    wrt.writerow(header)

# repeating the experiment 10x on different splits
for k in range(kn):

    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):

        X, y = ds

        print("\n\nGoing through " + sel_names[ds_cnt] + " " + str(k + 1) + " time")

        if load == "n":
            experiment_cv(n, wrt, X, y, k)
        else:
            if rerun:
                loaded = experiment_load(n, wrt, k, rerun)
            else:
                loaded = experiment_load(n, k=k, all_models=rerun)
            try:
                while True:
                    clf, the_200, X_hold, y_hold, selected, clf_name = loaded.__next__()
                    if not rerun:
                        send_noise(clf, clf_name, the_200, X, y, k, ds_cnt)
            except StopIteration:
                print("Loaded all")


if load == "n":
    f.close()

# experiment 5
