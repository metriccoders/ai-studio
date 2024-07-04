from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import OneClassSVM
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.svm import NuSVC
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, confusion_matrix, hamming_loss, jaccard_score, log_loss, matthews_corrcoef, balanced_accuracy_score, precision_recall_curve, roc_curve, zero_one_loss
from sklearn.model_selection import train_test_split


def get_kneighbors(n_neighbors, weights, algorithm, leaf_size, n_jobs):
    clf = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        leaf_size=leaf_size,
        n_jobs=n_jobs
    )

    return clf


def get_adaboost(n_estimators, learning_rate, algorithm, criterion, random_state):
    clf = AdaBoostClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        algorithm=algorithm,
        random_state=random_state
    )

    return clf


def get_bagging(
            n_estimators,
            max_samples,
            max_features,
            bootstrap,
            bootstrap_features,
            oob_score,
            warm_start,
            n_jobs,
            random_state
        ):
    clf = BaggingClassifier(
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=bootstrap,
        bootstrap_features=bootstrap_features,
        oob_score=oob_score,
        warm_start=warm_start,
        n_jobs=n_jobs,
        random_state=random_state
    )

    return clf

def get_bernoulli(alpha, force_alpha, binarize, fit_prior):
    clf = BernoulliNB(
        alpha=alpha,
        force_alpha=force_alpha,
        binarize=binarize,
        fit_prior=fit_prior
    )

    return clf


def get_calibrated_cv(method, cv, n_jobs):
    clf = CalibratedClassifierCV(
        method=method,
        cv=cv,
        n_jobs=n_jobs
    )

    return clf

def get_decision_tree(criterion,
                      splitter,
                      max_depth,
                      min_samples_split,
                      min_samples_leaf,
                      min_weight_fraction_leaf,
                      max_features,
                      max_leaf_nodes,
                      min_impurity_decrease, ccp_alpha):
    clf = DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        ccp_alpha=ccp_alpha
    )

    return clf


def get_extra_trees(n_estimators,
                    criterion,
                    max_depth,
                    min_samples_split,
                    min_samples_leaf,
                    min_weight_fraction_leaf,
                    max_features,
                    max_leaf_nodes,
                    min_impurity_decrease,
                    ccp_alpha,
                    bootstrap,
                    n_jobs,
                    random_state   ):
    clf = ExtraTreesClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        ccp_alpha=ccp_alpha,
        bootstrap=bootstrap,
        n_jobs=n_jobs,
        random_state=random_state
    )

    return clf


def get_gradient_boosting(criterion,
                          n_estimators,
                          max_depth,
                        min_samples_split,
                        min_samples_leaf,
                        min_weight_fraction_leaf,
                        max_features,
                        max_leaf_nodes,
                        min_impurity_decrease,
                        ccp_alpha,
                        random_state):
    clf = GradientBoostingClassifier(
        criterion=criterion,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        ccp_alpha=ccp_alpha,
        random_state=random_state
    )

    return clf


def get_random_forest(
                        n_estimators,
                        criterion,
                        max_depth,
                        min_samples_split,
                        min_samples_leaf,
                        min_weight_fraction_leaf,
                        max_features,
                        max_leaf_nodes,
                        min_impurity_decrease,
                                bootstrap,
                                n_jobs
                        ):
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        bootstrap=bootstrap,
        n_jobs=n_jobs
    )

    return clf

"""
def get_extra_tree(n_neighbors, weights, algorithm, leaf_size, n_jobs):
    clf = ExtraTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        random_state=random_state
    )

    return clf


def get_one_class_svm(n_neighbors, weights, algorithm, leaf_size, n_jobs):
    clf = OneClassSVM(
        kernel=kernel,
        gamma=gamma,
        coef0=coef0,
        cache_size=cache_size,
        shrinking=shrinking
    )

    return clf


def get_mlp(n_neighbors, weights, algorithm, leaf_size, n_jobs):
    clf = MLPClassifier(
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate=learning_rate,
        learning_rate_init=learning_rate_init,
        power_t=power_t,
        shuffle=shuffle,
        warm_start=warm_start,
        nesterovs_momentum=nesterovs_momentum
    )

    return clf


def get_radius_neighbors(n_neighbors, weights, algorithm, leaf_size, n_jobs):
    clf = RadiusNeighborsClassifier(
        radius=radius,
        weights=weights,
        algorithm=algorithm,
        leaf_size=leaf_size,
        outlier_label=outlier_label
    )

    return clf

def get_classifier_chain(n_neighbors, weights, algorithm, leaf_size, n_jobs):
    clf = ClassifierChain(
        chain_method=chain_method
    )

    return clf

def get_multi_output(n_neighbors, weights, algorithm, leaf_size, n_jobs):
    clf = MultiOutputClassifier(
        n_jobs=n_jobs
    )

    return clf

def get_outut_code(n_neighbors, weights, algorithm, leaf_size, n_jobs):
    clf = OutputCodeClassifier(
        n_jobs=n_jobs,
        random_state=random_state
    )

    return clf


def get_one_vs_one(n_neighbors, weights, algorithm, leaf_size, n_jobs):
    clf = OneVsOneClassifier(
        n_jobs=n_jobs
    )

    return clf


def get_one_vs_rest(n_neighbors, weights, algorithm, leaf_size, n_jobs):
    clf = OneVsRestClassifier(
        n_jobs=n_jobs
    )

    return clf


def get_sgd(n_neighbors, weights, algorithm, leaf_size, n_jobs):
    clf = SGDClassifier(
        loss=loss,
        penalty=penalty,
        alpha=alpha,
        fit_intercept=fit_intercept,
        max_iter=max_iter
    )

    return clf


def get_ridge_cv(n_neighbors, weights, algorithm, leaf_size, n_jobs):
    clf = RidgeClassifierCV(
        fit_intercept=fit_intercept
    )

    return clf


def get_ridge(n_neighbors, weights, algorithm, leaf_size, n_jobs):
    clf = RidgeClassifier(
        fit_intercept=fit_intercept,
        copy_X=copy_X,
        solver=solver,
        alpha=alpha
    )

    return clf

def get_passive_aggressive(n_neighbors, weights, algorithm, leaf_size, n_jobs):
    clf = PassiveAggressiveClassifier(
        fit_intercept=fit_intercept,
        max_iter=max_iter
    )

    return clf


def get_gaussian_process(n_neighbors, weights, algorithm, leaf_size, n_jobs):
    clf = GaussianProcessClassifier(
        optimizer=optimizer,
        n_restarts_optimizer=n_restarts_optimizer,
        max_iter_predict=max_iter_predict,
        warm_start=warm_start,
        copy_X_train=copy_X_train,
        random_state=random_state,
        multi_class=multi_class
    )

    return clf


def get_gaussian_nb(n_neighbors, weights, algorithm, leaf_size, n_jobs):

    clf = GaussianNB()
    return clf

def get_linear_discriminant(n_neighbors, weights, algorithm, leaf_size, n_jobs):
    clf = LinearDiscriminantAnalysis(
        solver=solver,
        store_covariance=store_covariance
    )

    return clf


def get_linear_svc(n_neighbors, weights, algorithm, leaf_size, n_jobs):
    clf = LinearSVC(
        penalty=penalty,
        loss=loss,
        fit_intercept=fit_intercept,
        random_state=random_state,
        max_iter=max_iter
    )

    return clf

def get_logistic_regr(n_neighbors, weights, algorithm, leaf_size, n_jobs):
    clf = LogisticRegression(
        penalty=penalty,
        dual=dual,
        solver=solver,
        fit_intercept=fit_intercept,
        random_state=random_state,
        max_iter=max_iter,
        warm_start=warm_start,
        multi_class=multi_class
    )

    return clf


def get_logistic_regr_cv(n_neighbors, weights, algorithm, leaf_size, n_jobs):
    clf = LogisticRegressionCV(
        Cs=cs,
        penalty=penalty,
        dual=dual,
        solver=solver,
        fit_intercept=fit_intercept,
        random_state=random_state,
        max_iter=max_iter,
        multi_class=multi_class
    )

    return clf


def get_multinomial(n_neighbors, weights, algorithm, leaf_size, n_jobs):

    clf = MultinomialNB(
        alpha=alpha,
        fit_prior=fit_prior
    )

    return clf

def get_nearest_centroid(n_neighbors, weights, algorithm, leaf_size, n_jobs):
    clf = NearestCentroid(
        metric=metric
    )

    return clf


def get_nu_svc(n_neighbors, weights, algorithm, leaf_size, n_jobs):
    clf = NuSVC(
        nu=nu,
        kernel=kernel,
        gamma=gamma,
        shrinking=shrinking,
        probability=probability,
        decision_function_shape=decision_function_shape
    )

    return clf


def get_perceptron(n_neighbors, weights, algorithm, leaf_size, n_jobs):
    clf = Perceptron(
        alpha=alpha,
        penalty=penalty,
        warm_start=warm_start,
        shuffle=shuffle,
        fit_intercept=fit_intercept,
        random_state=random_state,
        max_iter=max_iter
    )

    return clf

def get_quadratic_discriminant(n_neighbors, weights, algorithm, leaf_size, n_jobs):
    clf = QuadraticDiscriminantAnalysis(
        store_covariance=store_covariance
    )

    return clf

def get_svc(n_neighbors, weights, algorithm, leaf_size, n_jobs):
    clf = SVC(
        C=c,
        kernel=kernel,
        gamma=gamma,
        shrinking=shrinking,
        coef0=coef0,
        probability=probability
    )

    return clf

def get_gaussian_mixture(n_neighbors, weights, algorithm, leaf_size, n_jobs):
    clf = GaussianMixture(
        covariance_type=covariance_type,
        max_iter=max_iter,
        warm_start=warm_start
    )

    return clf
"""