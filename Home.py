import streamlit as st
import joblib
import zipfile
#from streamlit_lottie import st_lottie
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer, load_diabetes, load_linnerud
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
import io
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    r2_score,
    explained_variance_score
)
from sklearn.linear_model._bayes import ARDRegression
from sklearn.ensemble._weight_boosting import AdaBoostRegressor
from sklearn.ensemble._bagging import BaggingRegressor
from sklearn.linear_model._bayes import BayesianRidge
from sklearn.cross_decomposition._pls import CCA
from sklearn.tree._classes import DecisionTreeRegressor
from sklearn.linear_model._coordinate_descent import ElasticNet
from sklearn.linear_model._coordinate_descent import ElasticNetCV
from sklearn.tree._classes import ExtraTreeRegressor
from sklearn.ensemble._forest import ExtraTreesRegressor
from sklearn.linear_model._glm.glm import GammaRegressor
from sklearn.gaussian_process._gpr import GaussianProcessRegressor
from sklearn.ensemble._gb import GradientBoostingRegressor
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingRegressor
from sklearn.linear_model._huber import HuberRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors._regression import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model._least_angle import Lars
from sklearn.linear_model._least_angle import LarsCV
from sklearn.linear_model._coordinate_descent import Lasso
from sklearn.linear_model._coordinate_descent import LassoCV
from sklearn.linear_model._least_angle import LassoLars
from sklearn.linear_model._least_angle import LassoLarsCV
from sklearn.linear_model._least_angle import LassoLarsIC
from sklearn.linear_model._base import LinearRegression
from sklearn.svm._classes import LinearSVR
from sklearn.neural_network._multilayer_perceptron import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model._coordinate_descent import MultiTaskElasticNet
from sklearn.linear_model._coordinate_descent import MultiTaskElasticNetCV
from sklearn.linear_model._coordinate_descent import MultiTaskLasso
from sklearn.linear_model._coordinate_descent import MultiTaskLassoCV
from sklearn.svm._classes import NuSVR
from sklearn.linear_model._omp import OrthogonalMatchingPursuit
from sklearn.linear_model._omp import OrthogonalMatchingPursuitCV
from sklearn.cross_decomposition._pls import PLSCanonical
from sklearn.cross_decomposition._pls import PLSRegression
from sklearn.linear_model._passive_aggressive import PassiveAggressiveRegressor
from sklearn.linear_model._glm.glm import PoissonRegressor
from sklearn.linear_model._quantile import QuantileRegressor
from sklearn.linear_model._ransac import RANSACRegressor
from sklearn.neighbors._regression import RadiusNeighborsRegressor
from sklearn.ensemble._forest import RandomForestRegressor
from sklearn.multioutput import RegressorChain
from sklearn.linear_model._ridge import Ridge
from sklearn.linear_model._ridge import RidgeCV
from sklearn.linear_model._stochastic_gradient import SGDRegressor
from sklearn.svm._classes import SVR
from sklearn.linear_model._theil_sen import TheilSenRegressor
from sklearn.compose._target import TransformedTargetRegressor
from sklearn.linear_model._glm.glm import TweedieRegressor

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

requirements_txt_file = """
fastapi
numpy
"""

backend_api_python_program_content = """

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Define a request model
class PredictionRequest(BaseModel):
    # Sample POST Request: 
    # {
    #     "features": [5.1, 3.5, 1.4, 0.2]
    # }
    features: list[float]

# Initialize FastAPI app
app = FastAPI()

# Load the machine learning model
model = None

@app.on_event("startup")
def load_model():
    global model
    model = joblib.load("model.joblib")

@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Convert request features to a numpy array and reshape it for prediction
    features = np.array(request.features).reshape(1, -1)

    # Make the prediction
    prediction = model.predict(features)

    # Return the prediction result
    return {"prediction": prediction[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""


def create_zip(model_buffer):

    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, 'w') as zip_file:

        zip_file.writestr('backend_api.py', backend_api_python_program_content)

        zip_file.writestr('model.joblib', model_buffer.getvalue())

        zip_file.writestr('requirements.txt', requirements_txt_file)


    buffer.seek(0)
    return buffer



def main():
    st.markdown("<center><h1>Artificial Intelligence (AI) Studio</h1></center>", unsafe_allow_html=True)
    #st.lottie("https://lottie.host/f9ecc8cd-9a0e-49f5-bfbe-89bb59ca794b/Qnv20SfUVi.json", height=50, width=50, quality="high")
    st.markdown("<center><h4><b>By Metric Coders</b></h4></center>", unsafe_allow_html=True)
    dataset = load_iris()
    clf = AdaBoostClassifier()
    regr = ARDRegression()
    X = None
    y = None
    ml_algo_options = ["Classifiers", "Regressors"]

    algo_type = st.radio("Select the type of ML algorithm:", ml_algo_options)

    own_dataset = st.checkbox(label="Load Own Dataset (The CSV file should have the header by the name 'target' as the result column)", value=False)
    if own_dataset:
        uploaded_file = st.file_uploader("Upload Training Data in a CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)

            st.write("Preview of Dataset")
            st.write(data.head())

            X = data.drop("target", axis=1)
            y = data["target"]
    else:
        if algo_type == "Classifiers":
            dataset_option = st.selectbox("Dataset", ["Iris", "Digits", "Wine", "Breast Cancer"], index=0)
            if dataset_option == "Iris":
                dataset = load_iris()
            elif dataset_option == "Digits":
                dataset = load_digits()
            elif dataset_option == "Wine":
                dataset = load_wine()
            elif dataset_option == "Breast Cancer":
                dataset = load_breast_cancer()
        elif algo_type == "Regressors":
            dataset_option = st.selectbox("Dataset", ["Diabetes", "Linnerrud"], index=0)
            if dataset_option == "Diabetes":
                dataset = load_diabetes()
            elif dataset_option == "Linnerrud":
                dataset = load_linnerud()

        X = dataset.data
        y = dataset.target


    if algo_type == "Classifiers":
        ml_algorithm = st.selectbox("Classifiers", ["AdaBoost Classifier",
                                                 "Bagging Classifier",
                                                 "BernoulliNB",
                                                 "Calibrated Classifier CV",
                                                 "Decision Tree Classifier",
                                                 "Extra Trees Classifier",
                                                 "Gradient Boosting Classifier",
                                                 "KNeighbors Classifier",
                                                 "Random Forest Classifier",
                                                 "Extra Tree Classifier",
                                                 "One Class SVM",
                                                 "MLP Classifier",
                                                 "Radius Neighbors Classifier",
                                                 "Classifier Chain",
                                                 "Multi Output Classifier",
                                                 "Output Code Classifier",
                                                 "One Vs One Classifier",
                                                 "One Vs Rest Classifier",
                                                 "SGD Classifier",
                                                 "Ridge Classifier CV",
                                                 "Ridge Classifier",
                                                 "Passive Aggressive Classifier",
                                                 "Gaussian Process Classifier",
                                                 "Gaussian NB",
                                                 "Label Propagation",
                                                 "Label Spreading",
                                                 "Linear Discriminant Analysis",
                                                 "Linear SVC",
                                                 "Logistic Regression",
                                                 "Logistic Regression CV",
                                                 "Multinomial NB",
                                                 "Nearest Centroid",
                                                 "NuSVC",
                                                 "Perceptron",
                                                 "Quadratic Discriminant Analysis",
                                                 "SVC",
                                                 "Gaussian Mixture",
                                                 ], index=0)
    else:
        ml_algorithm = st.selectbox("Regressors", [
            "ARD Regression",
            "AdaBoost Regressor",
            "Bagging Regressor",
            "Bayesian Ridge",
            "CCA",
            "Decision Tree Regressor",
            "Elastic Net",
            "Elastic Net CV",
            "Extra Tree Regressor",
            "Extra Trees Regressor",
            "Gamma Regressor",
            "Gaussian Process Regressor",
            "Gradient Boosting Regressor",
            "Hist Gradient Boosting Regressor",
            "Huber Regressor",
            "Isotonic Regression",
            "KNeighbors Regressor",
            "Kernel Ridge",
            "Lars",
            "Lars CV",
            "Lasso",
            "Lasso CV",
            "Lasso Lars",
            "Lasso Lars CV",
            "Lasso Lars IC",
            "Linear Regression",
            "Linear SVR",
            "MLP Regressor",
            "Multi Output Regressor",
            "Multi Task Elastic Net",
            "Multi Task Elastic Net CV",
            "Multi Task Lasso",
            "Multi Task Lasso CV",
            "NuSVR",
            "Orthogonal Matching Pursuit",
            "Orthogonal Matching Pursuit CV",
            "PLS Canonical",
            "PLS Regression",
            "Passive Aggressive Regressor",
            "Poisson Regressor",
            "Quantile Regressor",
            "RANSAC Regressor",
            "Radius Neighbors Regressor",
            "Random Forest Regressor",
            "Regressor Chain",
            "Ridge",
            "Ridge CV",
            "SGD Regressor",
            "SVR",
            "Theil Sen Regressor",
            "Transformed Target Regressor",
            "TweedieRegressor"
        ], index=0)
    col1, col2 = st.columns(2)

    col1.markdown("<center><h3>Hyperparameters</h3></center>" ,unsafe_allow_html=True)
    if ml_algorithm == "KNeighbors Classifier":
        n_neighbors = col1.slider("n_neighbors", min_value=1, max_value=100, value=5, step=1)
        weights = col1.selectbox("weights", ["uniform", "distance"], index=0)
        algorithm = col1.selectbox("algorithm", ["auto", "ball_tree", "kd_tree", "brute"], index=0)

        leaf_size = col1.slider("leaf_size", min_value=1, max_value=100, value=30, step=1)
        n_jobs = col1.slider("n_jobs", min_value=1, max_value=100, value=5, step=1)

        clf = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
        )
    elif ml_algorithm == "AdaBoost Classifier":
        n_estimators = col1.slider("n_estimators", min_value=1, max_value=2000, value=50, step=1)
        learning_rate = col1.slider("learning_rate", min_value=1.0, max_value=10.0, value=1.0, step=0.1)
        algorithm = col1.selectbox("algorithm", ["SAMME", "SAMME.R"], index=0)
        criterion = col1.selectbox("criterion", ["gini", "entropy", "log_loss"], placeholder="Choose a criterion")
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)
        clf = AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=random_state
        )

    elif ml_algorithm == "Bagging Classifier":
        n_estimators = col1.slider("n_estimators", min_value=1, max_value=2000, value=10, step=1)
        max_samples = col1.slider("max_samples", min_value=1, max_value=10, value=1, step=1)
        max_features = col1.slider("max_features", min_value=1.0, max_value=5.0, value=0.1)
        bootstrap = col1.selectbox("bootstrap", [True, False], index=0)
        bootstrap_features = col1.selectbox("bootstrap_features", [True, False], index=0)
        oob_score = col1.selectbox("oob_score", [True, False], index=0)
        warm_start = col1.selectbox("warm_start", [True, False], index=1)
        n_jobs = col1.slider("n_jobs", min_value=1, max_value=10, value=1, step=1)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)

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

    elif ml_algorithm == "BernoulliNB":
        alpha = col1.slider("alpha", min_value=1.0, max_value=10.0, value=1.0, step=0.1)
        force_alpha = col1.selectbox("force_alpha", [True, False], index=0)
        binarize = col1.slider("binarize", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        fit_prior = col1.selectbox("fit_prior", [True, False], index=0)

        clf = BernoulliNB(
            alpha=alpha,
            force_alpha=force_alpha,
            binarize=binarize,
            fit_prior=fit_prior
        )

    elif ml_algorithm == "Calibrated Classifier CV":
        method = col1.selectbox("method", ["sigmoid", "istonic"], index=0)
        cv = col1.slider("cv", min_value=2, max_value=10, value=2, step=1)
        n_jobs = col1.slider("n_jobs", min_value=1, max_value=10, value=5, step=1)
        clf = CalibratedClassifierCV(
            method=method,
            cv=cv,
            n_jobs=n_jobs
        )

    elif ml_algorithm == "Decision Tree Classifier":
        criterion = col1.selectbox("criterion", ["gini", "entropy", "log_loss"], placeholder="Choose a criterion", index=0)
        splitter = col1.selectbox("splitter", ["best", "random"], index=0)
        max_depth = col1.slider("max_depth", min_value=3, max_value=200, value=3, step=1)
        min_samples_split = col1.slider("min_samples_split", min_value=2, max_value=100, value=2, step=1)
        min_samples_leaf = col1.slider("min_samples_leaf", min_value=1, max_value=200, value=1, step=1)
        min_weight_fraction_leaf = col1.slider("min_weight_fraction_leaf", min_value=0.0, max_value=0.5, value=0.0,
                                               step=0.1)
        max_features = col1.selectbox("max_features", ["sqrt", "log2", None], placeholder="Choose max. number of features",
                                      index=2)
        ccp_alpha = col1.slider("ccp_alpha", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        max_leaf_nodes = col1.slider("max_leaf_nodes", min_value=2, max_value=100, value=2, step=1)
        min_impurity_decrease = col1.slider("min_impurity_decrease", min_value=0.0, max_value=3.0, value=0.0, step=0.1)

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

    elif ml_algorithm == "Extra Trees Classifier":
        n_estimators = col1.slider("n_estimators", min_value=10, max_value=300, value=100, step=1)
        criterion = col1.selectbox("criterion", ["gini", "entropy", "log_loss"], placeholder="Choose a criterion", index=0)
        max_depth = col1.slider("max_depth", min_value=3, max_value=200, value=3, step=1)
        min_samples_split = col1.slider("min_samples_split", min_value=2, max_value=100, value=2, step=1)
        min_samples_leaf = col1.slider("min_samples_leaf", min_value=1, max_value=200, value=1, step=1)
        min_weight_fraction_leaf = col1.slider("min_weight_fraction_leaf", min_value=0.0, max_value=0.5, value=0.0,
                                               step=0.1)
        max_features = col1.selectbox("max_features", ["sqrt", "log2", None], placeholder="Choose max. number of features",
                                      index=0)
        ccp_alpha = col1.slider("ccp_alpha", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        max_leaf_nodes = col1.slider("max_leaf_nodes", min_value=2, max_value=100, value=2, step=1)
        min_impurity_decrease = col1.slider("min_impurity_decrease", min_value=0.0, max_value=3.0, value=0.0, step=0.1)
        bootstrap = col1.selectbox("bootstrap", [False, True], placeholder="Select bootstrap", index=0)
        n_jobs = col1.slider("slider", min_value=1, max_value=10, value=1, step=1)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)

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

    elif ml_algorithm == "Gradient Boosting Classifier":
        n_estimators = col1.slider("n_estimators", min_value=10, max_value=300, value=100, step=1)
        loss = col1.selectbox("loss", ["log_loss", "exponential"], placeholder="Select loss", index=0)
        criterion = col1.selectbox("criterion", ["friedman_mse", "squared_error"], placeholder="Choose a criterion",
                                   index=0)
        max_depth = col1.slider("max_depth", min_value=3, max_value=200, value=3, step=1)
        min_samples_split = col1.slider("min_samples_split", min_value=2, max_value=100, value=2, step=1)
        min_samples_leaf = col1.slider("min_samples_leaf", min_value=1, max_value=200, value=1, step=1)
        min_weight_fraction_leaf = col1.slider("min_weight_fraction_leaf", min_value=0.0, max_value=0.5, value=0.0,
                                               step=0.1)
        max_features = col1.selectbox("max_features", ["sqrt", "log2", None], placeholder="Choose max. number of features",
                                      index=0)
        ccp_alpha = col1.slider("ccp_alpha", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        max_leaf_nodes = col1.slider("max_leaf_nodes", min_value=2, max_value=100, value=2, step=1)
        min_impurity_decrease = col1.slider("min_impurity_decrease", min_value=0.0, max_value=3.0, value=0.0, step=0.1)
        n_jobs = col1.slider("slider", min_value=1, max_value=10, value=1, step=1)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)

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

    elif ml_algorithm == "KNeighbors Classifier":
        n_neighbors = col1.slider("n_neighbors", min_value=1, max_value=100, value=5, step=1)
        weights = col1.selectbox("weights", ["uniform", "distance"], index=0)
        algorithm = col1.selectbox("algorithm", ["auto", "ball_tree", "kd_tree", "brute"], index=0)

        leaf_size = col1.slider("leaf_size", min_value=1, max_value=100, value=30, step=1)
        n_jobs = col1.slider("n_jobs", min_value=1, max_value=100, value=5, step=1)

        clf = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,

        )

    elif ml_algorithm == "Random Forest Classifier":

        n_estimators = col1.slider("n_estimators", min_value=1, max_value=2000, value=100, step=1)
        criterion = col1.selectbox("criterion", ["gini", "entropy", "log_loss"], placeholder="Choose a criterion")
        max_depth = col1.slider("max_depth", min_value=3, max_value=200, value=3, step=1)
        min_samples_split = col1.slider("min_samples_split", min_value=2, max_value=100, value=2, step=1)
        min_samples_leaf = col1.slider("min_samples_leaf", min_value=1, max_value=200, value=1, step=1)
        min_weight_fraction_leaf = col1.slider("min_weight_fraction_leaf", min_value=0.0, max_value=0.5, value=0.0,
                                               step=0.1)
        max_features = col1.selectbox("max_features", ["sqrt", "log2", None], placeholder="Choose max. number of features")
        max_leaf_nodes = col1.slider("max_leaf_nodes", min_value=2, max_value=100, value=2, step=1)
        min_impurity_decrease = col1.slider("min_impurity_decrease", min_value=0.0, max_value=3.0, value=0.0, step=0.1)
        bootstrap = col1.checkbox("bootstrap", True)
        n_jobs = col1.slider("n_jobs", min_value=5, max_value=100, value=5, step=1)

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

    elif ml_algorithm == "Extra Tree Classifier":

        criterion = col1.selectbox("criterion", ["gini", "entropy", "log_loss"], placeholder="Choose a criterion", index=0)
        splitter = col1.selectbox("splitter", ["random", "best"], placeholder="Choose a splitter", index=0)
        max_depth = col1.slider("max_depth", min_value=3, max_value=200, value=3, step=1)
        min_samples_split = col1.slider("min_samples_split", min_value=2, max_value=100, value=2, step=1)
        min_samples_leaf = col1.slider("min_samples_leaf", min_value=1, max_value=200, value=1, step=1)
        min_weight_fraction_leaf = col1.slider("min_weight_fraction_leaf", min_value=0.0, max_value=0.5, value=0.0,
                                               step=0.1)
        max_features = col1.selectbox("max_features", ["sqrt", "log2", None], placeholder="Choose max. number of features", index=0)
        max_leaf_nodes = col1.slider("max_leaf_nodes", min_value=2, max_value=100, value=2, step=1)
        min_impurity_decrease = col1.slider("min_impurity_decrease", min_value=0.0, max_value=3.0, value=0.0, step=0.1)
        bootstrap = col1.checkbox("bootstrap", True)
        n_jobs = col1.slider("n_jobs", min_value=5, max_value=100, value=5, step=1)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)

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


    elif ml_algorithm == "One Class SVM":

        kernel = col1.selectbox("kernel", ["linear", "poly", "rbf", "sigmoid", "precomputed"], index=2)
        gamma = col1.selectbox("gamma", ["scale", "auto"], index=0)

        coef0 = col1.slider("coef0", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
        cache_size = col1.slider("cache_size", min_value=100, max_value=1000, value=200, step=100)
        shrinking = col1.selectbox("shrinking", [True, False], index=0)


        clf = OneClassSVM(
            kernel=kernel,
            gamma=gamma,
            coef0=coef0,
            cache_size=cache_size,
            shrinking=shrinking
        )

    elif ml_algorithm == "MLP Classifier":
        activation = col1.selectbox("activation", ["identity", "logistic", "tanh", "relu"], index=3)
        solver = col1.selectbox("solver", ["lbfgs", "sgd", "adam"], index=2)
        alpha = col1.slider("alpha", min_value=0.0001, max_value=1, value=0.0001, step=0.0001)
        learning_rate = col1.selectbox("learning_rate", ["constant", "invscaling", "adaptive"], index=0)

        learning_rate_init = col1.slider("learning_rate_init", min_value=0.001, max_value=1, value=0.001, step=0.001)
        power_t = col1.slider("power_t", min_value=0.1, max_value=10, value=0.5, step=0.1)
        shuffle = col1.selectbox("shuffle", [True, False], index=0)
        warm_start = col1.selectbox("warm_start", [True, False], index=0)
        nesterovs_momentum = col1.selectbox("nesterovs_momentum", [True, False], index=0)

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


    elif ml_algorithm == "Radius Neighbors Classifier":
        radius = col1.slider("radius", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        weights = col1.selectbox("weights", ["uniform", "distance"], index=0)
        algorithm = col1.selectbox("algorithm", ["auto", "ball_tree", "kd_tree", "brute"], index=0)
        leaf_size = col1.slider("leaf_size", min_value=10, max_value=100, value=30, step=1)
        outlier_label = col1.selectbox("outlier_label", ["manual label", "most_frequent", None], index=2)

        clf = RadiusNeighborsClassifier(
            radius=radius,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            outlier_label=outlier_label
        )

    elif ml_algorithm == "Classifier Chain":
        chain_method = col1.selectbox("chain_method", ["predict", "predict_proba", "predict_log_proba", "decision_function"], index=0)
        clf = ClassifierChain(
            chain_method=chain_method
        )

    elif ml_algorithm == "Multi Output Classifier":

        n_jobs = col1.slider("n_jobs", min_value=5, max_value=100, value=5, step=1)

        clf = MultiOutputClassifier(
            n_jobs=n_jobs
        )


    elif ml_algorithm == "Output Code Classifier":
        n_jobs = col1.slider("n_jobs", min_value=5, max_value=100, value=5, step=1)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)

        clf = OutputCodeClassifier(
            n_jobs=n_jobs,
            random_state=random_state
        )

    elif ml_algorithm == "One Vs One Classifier":

        n_jobs = col1.slider("n_jobs", min_value=5, max_value=100, value=5, step=1)

        clf = OneVsOneClassifier(
            n_jobs=n_jobs
        )


    elif ml_algorithm == "One Vs Rest Classifier":

        n_jobs = col1.slider("n_jobs", min_value=5, max_value=100, value=5, step=1)

        clf = OneVsRestClassifier(
            n_jobs=n_jobs
        )


    elif ml_algorithm == "SGD Classifier":

        loss = col1.selectbox("loss", ["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron", "squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"], index=0)
        penalty = col1.selectbox("penalty",
                              ["l2", "l1", "elasticnet", None], index=0)

        alpha = col1.slider("alpha", min_value=0.0001, max_value=1, value=0.0001, step=0.0001)
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        max_iter = col1.slider("max_iter", min_value=1000, max_value=100000, value=1000, step=1000)

        clf = SGDClassifier(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter
        )


    elif ml_algorithm == "Ridge Classifier CV":
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)

        clf = RidgeClassifierCV(
            fit_intercept=fit_intercept
        )


    elif ml_algorithm == "Ridge Classifier":
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        copy_X = col1.selectbox("copy_X", [True, False], index=0)
        solver = col1.selectbox("solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"], index=0)
        alpha = col1.slider("alpha", min_value=1.0, max_value=100.0, value=1.0, step=1.0)

        clf = RidgeClassifier(
        fit_intercept=fit_intercept,
            copy_X=copy_X,
            solver=solver,
            alpha=alpha
        )


    elif ml_algorithm == "Passive Aggressive Classifier":
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        max_iter = col1.slider("max_iter", min_value=1000, max_value=10000, value=1000, step=100)

        clf = PassiveAggressiveClassifier(
            fit_intercept=fit_intercept,
            max_iter=max_iter
        )


    elif ml_algorithm == "Gaussian Process Classifier":

        optimizer = col1.selectbox("optimizer", ["fmin_l_bfgs_b", None], index=0)
        n_restarts_optimizer = col1.slider("n_restarts_optimizer", min_value=0, max_value=200, value=0, step=1)
        max_iter_predict = col1.slider("max_iter_predict", min_value=10, max_value=1000, value=100, step=10)
        warm_start = col1.selectbox("warm_start", [True, False], index=1)
        copy_X_train = col1.selectbox("copy_X_train", [True, False], index=0)
        random_state = col1.slider("random_state", min_value=1, max_value=100, random_state=42, value=1)
        multi_class = col1.selectbox("multi_class", ["one_vs_rest", "one_vs_one"], index=0)

        clf = GaussianProcessClassifier(
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            max_iter_predict=max_iter_predict,
            warm_start=warm_start,
            copy_X_train=copy_X_train,
            random_state=random_state,
            multi_class=multi_class
        )


    elif ml_algorithm == "Gaussian NB":
        clf = GaussianNB()

    elif ml_algorithm == "Linear Discriminant Analysis":
        solver = col1.selectbox("solver", ["svd", "lsqr", "eigen"], index=0)
        store_covariance = col1.selectbox("store_covariance", [True, False], index=0)

        clf = LinearDiscriminantAnalysis(
            solver=solver,
            store_covariance=store_covariance
        )


    elif ml_algorithm == "Linear SVC":
        penalty = col1.selectbox("penalty", ["l1", "l2"], index=1)
        loss = col1.selectbox("loss", ["hinge", "squared_hinge"], index=1)
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)
        max_iter = col1.slider("max_iter", min_value=1000, max_value=10000, value=1000, step=100)

        clf = LinearSVC(
            penalty=penalty,
            loss=loss,
            fit_intercept=fit_intercept,
            random_state=random_state,
            max_iter=max_iter
        )

    elif ml_algorithm == "Logistic Regression":

        penalty = col1.selectbox("penalty", ["l1", "l2", "elasticnet", None], index=1)
        dual = col1.selectbox("dual", [True, False], index=1)
        solver = col1.selectbox("solver", ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"], index=0)
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)
        max_iter = col1.slider("max_iter", min_value=1000, max_value=10000, value=1000, step=100)
        warm_start = col1.selectbox("warm_start", [True, False], index=1)
        multi_class = col1.selectbox("multi_class", ["auto", "ovr", "multinomial"], index=0)

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

    elif ml_algorithm == "Logistic Regression CV":
        cs = col1.slider("Cs", min_value=1, max_value=100, value=10, step=1)
        penalty = col1.selectbox("penalty", ["l1", "l2", "elasticnet"], index=1)
        dual = col1.selectbox("dual", [True, False], index=1)
        solver = col1.selectbox("solver", ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"], index=0)
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)
        max_iter = col1.slider("max_iter", min_value=1000, max_value=10000, value=1000, step=100)
        warm_start = col1.selectbox("warm_start", [True, False], index=1)
        multi_class = col1.selectbox("multi_class", ["auto", "ovr", "multinomial"], index=0)

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

    elif ml_algorithm == "Multinomial NB":

        alpha = col1.selectbox("alpha", [True, False], index=0)
        fit_prior = col1.selectbox("fit_prior", [True, False], index=0)
        clf = MultinomialNB(
            alpha=alpha,
            fit_prior=fit_prior
        )

    elif ml_algorithm == "Nearest Centroid":
        metric = col1.selectbox("metric", ["euclidean", "manhattan"], index=0)

        clf = NearestCentroid(
            metric=metric
        )


    elif ml_algorithm == "NuSVC":

        nu = col1.slider("nu", min_value=0.1, max_value=10.0, value=0.5, step=0.1)
        kernel = col1.selectbox("kernel", ["linear", "poly", "rbf", "sigmoid", "precomputed"], index=2)
        gamma = col1.selectbox("gamma", ['scale', 'auto'], index=0)
        shrinking = col1.selectbox("shrinking", [True, False], index=0)
        probability = col1.selectbox("probability", [True, False], index=1)
        decision_function_shape = col1.slider("decision_function_shape", ['ovo', 'ovr'], index=1)

        clf = NuSVC(
            nu=nu,
            kernel=kernel,
            gamma=gamma,
            shrinking=shrinking,
            probability=probability,
            decision_function_shape=decision_function_shape
        )
    elif ml_algorithm == "Perceptron":

        alpha = col1.slider("alpha", min_value=0.0001, max_value=10.000, value=0.0001, step=0.1)
        penalty = col1.selectbox("penalty", ["l1", "l2", "elasticnet", None], index=3)
        warm_start = col1.selectbox("warm_start", [True, False], index=1)
        shuffle = col1.selectbox("shuffle", [True, False], index=0)
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)
        max_iter = col1.slider("max_iter", min_value=1000, max_value=10000, value=1000, step=100)

        clf = Perceptron(
            alpha=alpha,
            penalty=penalty,
            warm_start=warm_start,
            shuffle=shuffle,
            fit_intercept=fit_intercept,
            random_state=random_state,
            max_iter=max_iter
        )

    elif ml_algorithm == "Quadratic Discriminant Analysis":

        store_covariance = col1.selectbox("store_covariance", [True, False], index=1)
        clf = QuadraticDiscriminantAnalysis(
            store_covariance=store_covariance
        )



    elif ml_algorithm == "SVC":

        c = col1.slider("C", min_value=1.0, max_value=20.0, value=1.0, step=0.1)
        kernel = col1.selectbox("kernel", ["linear", "poly", "rbf", "sigmoid", "precomputed"], index=2)
        gamma = col1.selectbox("gamma", ["auto", "scale"], index=1)
        shrinking = col1.selectbox("shrinking", [True, False], index=0)
        coef0 = col1.slider("coef0", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        probability = col1.selectbox("probability", [True, False], index=1)

        clf = SVC(
            C=c,
            kernel=kernel,
            gamma=gamma,
            shrinking=shrinking,
            coef0=coef0,
            probability=probability
        )


    elif ml_algorithm == "Gaussian Mixture":
        covariance_type = col1.selectbox("covariance_type", ["full", "tied", "diag", "spherical"], index=0)
        max_iter = col1.slider("max_iter", min_value=100, max_value=10000, value=100, step=100)
        warm_start = col1.selectbox("warm_start", [True, False], index=1)
        clf = GaussianMixture(
            covariance_type=covariance_type,
            max_iter=max_iter,
            warm_start=warm_start
        )


    elif ml_algorithm == "ARD Regression":
        compute_score = col1.selectbox("compute_score", [True, False], index=1)
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        copy_X = col1.selectbox("copy_X", [True, False], index=0)
        regr = ARDRegression(
            compute_score=compute_score,
            fit_intercept=fit_intercept,
            copy_X=copy_X
        )

    elif ml_algorithm == "AdaBoost Regressor":
        loss = col1.selectbox("loss", ["linear", "square", "exponential"], index=0)
        n_estimators = col1.slider("n_estimators", min_value=10, max_value=500, value=50, step=10)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)

        regr = AdaBoostRegressor(
            loss=loss,
            n_estimators=n_estimators,
            random_state=random_state
        )
    elif ml_algorithm == "Bagging Regressor":
        n_estimators = col1.slider("n_estimators", min_value=10, max_value=500, value=10, step=10)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)
        bootstrap = col1.selectbox("bootstrap", [True, False], index=0)
        bootstrap_features = col1.selectbox("bootstrap_features", [True, False], index=1)
        oob_score = col1.selectbox("oob_score", [True, False], index=1)
        warm_state = col1.selectbox("warm_state", [True, False], index=1)

        regr = BaggingRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_state
        )

    elif ml_algorithm == "Bayesian Ridge":

        compute_score = col1.selectbox("compute_score", [True, False], index=1)
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        copy_X = col1.selectbox("copy_X", [True, False], index=0)

        regr = BayesianRidge(
            compute_score=compute_score,
            fit_intercept=fit_intercept,
            copy_X=copy_X
        )

    elif ml_algorithm == "CCA":
        n_components = col1.slider("n_components", min_value=1, max_value=100, value=2, step=1)
        scale = col1.selectbox("scale", [True, False], index=0)
        max_iter = col1.slider("max_iter", min_value=100, max_value=1000, value=500, step=10)
        copy = col1.selectbox("copy", [True, False], index=0)

        regr = CCA(
            n_components=n_components,
            scale=scale,
            max_iter=max_iter,
            copy=copy
        )

    elif ml_algorithm == "Decision Tree Regressor":
        criterion = col1.selectbox("criterion", ["squared_error", "friedman_mse", "absolute_error", "poisson"], index=0)
        splitter = col1.selectbox("splitter", ["best", "random"], index=0)

        regr = DecisionTreeRegressor(
            criterion=criterion,
            splitter=splitter
        )
    elif ml_algorithm == "Elastic Net":
        alpha = col1.slider("alpha", min_value=1.0, max_value=10.0, value=1.0, step=0.1)
        l1_ratio = col1.slider("l1_ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        max_iter = col1.slider("max_iter", min_value=1000, max_value=10000, value=1000, step=100)
        copy_X = col1.selectbox("copy_X", [True, False], index=0)
        warm_start = col1.selectbox("warm_start", [True, False], index=1)
        positive = col1.selectbox("positive", [True, False], index=1)
        selection = col1.selectbox("selection", ['cyclic', 'random'], index=0)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)

        regr = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            copy_X=copy_X,
            warm_start=warm_start,
            positive=positive,
            selection=selection,
            random_state=random_state
        )

    elif ml_algorithm == "Elastic Net CV":

        l1_ratio = col1.slider("l1_ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        max_iter = col1.slider("max_iter", min_value=1000, max_value=10000, value=1000, step=100)
        copy_X = col1.selectbox("copy_X", [True, False], index=0)

        positive = col1.selectbox("positive", [True, False], index=1)
        selection = col1.selectbox("selection", ['cyclic', 'random'], index=0)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)

        regr = ElasticNetCV(
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            copy_X=copy_X,
            positive=positive,
            selection=selection,
            random_state=random_state
        )

    elif ml_algorithm == "Extra Tree Regressor":

        criterion = col1.selectbox("criterion", ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'], index=0)
        splitter = col1.selectbox("splitter", ['random', 'best'], index=0)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)

        regr = ExtraTreeRegressor(
            criterion=criterion,
            splitter=splitter,
            random_state=random_state
        )

    elif ml_algorithm == "Extra Trees Regressor":

        criterion = col1.selectbox("criterion", ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'], index=0)
        bootstrap = col1.selectbox("bootstrap", [True, False], index=1)
        oob_score = col1.selectbox("oob_score", [True, False], index=1)
        warm_state = col1.selectbox("warm_state", [True, False], index=1)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)

        regr = ExtraTreesRegressor(
            criterion=criterion,
            bootstrap=bootstrap,
            oob_score=oob_score,
            warm_start=warm_state,
            random_state=random_state
        )

    elif ml_algorithm == "Gamma Regressor":
        alpha = col1.slider("alpha", min_value=1, max_value=100, value=1, step=1)
        solver = col1.selectbox("solver", ['lbfgs', 'newton-cholesky'], index=0)
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=1)
        warm_state = col1.selectbox("warm_state", [True, False], index=1)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)

        regr = GammaRegressor(
            alpha=alpha,
            solver=solver,
            fit_intercept=fit_intercept,
            warm_start=warm_state,
            random_state=random_state
        )

    elif ml_algorithm == "Gaussian Process Regressor":
        normalize_y = col1.selectbox("normalize_y", [True, False], index=1)

        regr = GaussianProcessRegressor(
            normalize_y=normalize_y
        )

    elif ml_algorithm == "Gradient Boosting Regressor":
        loss = col1.selectbox("loss", ['squared_error', 'absolute_error', 'huber', 'quantile'], index=0)
        learning_rate = col1.slider("learning_rate", min_value=0.1, max_value=10.0, value=0.1, step=0.1)
        n_estimators = col1.slider("n_estimators", min_value=100, max_value=1000, value=100, step=10)
        criterion = col1.selectbox("criterion", ['friedman_mse', 'squared_error'], index=0)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)
        warm_state = col1.selectbox("warm_state", [True, False], index=1)

        regr = GradientBoostingRegressor(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            criterion=criterion,
            random_state=random_state,
            warm_start=warm_state
        )

    elif ml_algorithm == "Hist Gradient Boosting Regressor":
        loss = col1.selectbox("loss", ['squared_error', 'absolute_error', 'gamma', 'poisson', 'quantile'], index=0)
        learning_rate = col1.slider("learning_rate", min_value=0.1, max_value=10.0, value=0.1, step=0.1)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)
        warm_state = col1.selectbox("warm_state", [True, False], index=1)
        max_iter = col1.slider("max_iter", min_value=100, max_value=1000, value=100, step=10)


        regr = HistGradientBoostingRegressor(
            loss=loss,
            learning_rate=learning_rate,
            max_iter=max_iter,
            random_state=random_state,
            warm_start=warm_state
        )

    elif ml_algorithm == "Huber Regressor":
        epsilon = col1.slider("epsilon", min_value=1.0, max_value=100.0, value=1.35, step=0.01)
        alpha = col1.slider("alpha", min_value=0.0001, max_value=1.0, value=0.0001, step=0.0001)
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        warm_state = col1.selectbox("warm_state", [True, False], index=0)
        max_iter = col1.slider("max_iter", min_value=100, max_value=1000, value=100, step=10)


        regr = HuberRegressor(
            epsilon=epsilon,
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            warm_start=warm_state
        )

    elif ml_algorithm == "Isotonic Regression":
        increasing = col1.selectbox("increasing", [True, False], index=0)

        regr = IsotonicRegression(
            increasing=increasing
        )

    elif ml_algorithm == "KNeighbors Regressor":
        n_neighbors = col1.slider("n_neighbors", min_value=1, max_value=100, value=5, step=1)
        algorithm = col1.selectbox("algorithm", ['auto', 'ball_tree', 'kd_tree', 'brute'], index=0)
        leaf_size = col1.slider("leaf_size", min_value=10, max_value=100, value=30, step=1)

        regr = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size
        )

    elif ml_algorithm == "Kernel Ridge":
        degree = col1.slider("degree", min_value=1, max_value=100, value=3, step=1)

        regr = KernelRidge(
            degree=degree
        )

    elif ml_algorithm == "Lars":
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        copy_X = col1.selectbox("copy_X", [True, False], index=0)
        fit_path = col1.selectbox("fit_path", [True, False], index=0)

        regr = Lars(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            fit_path=fit_path
        )

    elif ml_algorithm == "Lars CV":
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        copy_X = col1.selectbox("copy_X", [True, False], index=0)
        max_iter = col1.slider("max_iter", min_value=100, max_value=2000, value=500, step=100)

        regr = LarsCV(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter
        )

    elif ml_algorithm == "Lasso":
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        copy_X = col1.selectbox("copy_X", [True, False], index=0)
        warm_start = col1.selectbox("warm_start", [True, False], index=1)
        max_iter = col1.slider("max_iter", min_value=100, max_value=2000, value=500, step=100)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)

        regr = Lasso(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter,
            warm_start=warm_start,
            random_state=random_state
        )

    elif ml_algorithm == "Lasso CV":
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        copy_X = col1.selectbox("copy_X", [True, False], index=0)
        positive = col1.selectbox("positive", [True, False], index=1)
        max_iter = col1.slider("max_iter", min_value=100, max_value=2000, value=500, step=100)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)

        regr = LassoCV(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter,
            random_state=random_state,
            positive=positive
        )

    elif ml_algorithm == "Lasso Lars":
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        copy_X = col1.selectbox("copy_X", [True, False], index=0)
        positive = col1.selectbox("positive", [True, False], index=1)
        fit_path = col1.selectbox("fit_path", [True, False], index=0)
        max_iter = col1.slider("max_iter", min_value=100, max_value=2000, value=500, step=100)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)

        regr = LassoLars(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter,
            random_state=random_state,
            positive=positive,
            fit_path=fit_path
        )

    elif ml_algorithm == "Lasso Lars CV":
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        copy_X = col1.selectbox("copy_X", [True, False], index=0)
        positive = col1.selectbox("positive", [True, False], index=1)
        max_iter = col1.slider("max_iter", min_value=100, max_value=2000, value=500, step=100)

        regr = LassoLarsCV(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter,
            positive=positive,
        )

    elif ml_algorithm == "Lasso Lars IC":
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        copy_X = col1.selectbox("copy_X", [True, False], index=0)
        positive = col1.selectbox("positive", [True, False], index=1)
        max_iter = col1.slider("max_iter", min_value=100, max_value=2000, value=500, step=100)

        regr = LassoLarsIC(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter,
            positive=positive,
        )

    elif ml_algorithm == "Linear Regression":
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        copy_X = col1.selectbox("copy_X", [True, False], index=0)
        positive = col1.selectbox("positive", [True, False], index=1)

        regr = LinearRegression(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            positive=positive,
        )

    elif ml_algorithm == "Linear SVR":
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        loss = col1.selectbox("loss", ['epsilon_insensitive', 'squared_epsilon_insensitive'], index=0)
        max_iter = col1.slider("max_iter", min_value=1000, max_value=5000, value=1000, step=100)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)

        regr = LinearSVR(
            fit_intercept=fit_intercept,
            loss=loss,
            max_iter=max_iter,
            random_state=random_state
        )

    elif ml_algorithm == "MLP Regressor":
        activation = col1.selectbox("activation", ['identity', 'logistic', 'tanh', 'relu'], index=3)
        solver = col1.selectbox("solver", ['lbfgs', 'sgd', 'adam'], index=2)
        learning_rate = col1.selectbox("learning_rate", ['constant', 'invscaling', 'adaptive'], index=0)
        max_iter = col1.slider("max_iter", min_value=100, max_value=5000, value=200, step=100)
        shuffle = col1.selectbox("shuffle", [True, False], index=0)
        warm_start = col1.selectbox("warm_start", [True, False], index=1)
        nesterovs_momentum = col1.selectbox("nesterovs_momentum", [True, False], index=0)
        early_stopping = col1.selectbox("early_stopping", [True, False], index=1)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)

        regr = MLPRegressor(
            activation=activation,
            solver=solver,
            learning_rate=learning_rate,
            max_iter=max_iter,
            shuffle=shuffle,
            warm_start=warm_start,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            random_state=random_state
            )

    elif ml_algorithm == "Multi Output Regressor":

        regr = MultiOutputRegressor()


    elif ml_algorithm == "Multi Task Elastic Net":
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        copy_X = col1.selectbox("copy_X", [True, False], index=0)
        warm_start = col1.selectbox("warm_start", [True, False], index=0)
        max_iter = col1.slider("max_iter", min_value=1000, max_value=10000, value=1000, step=100)
        selection = col1.selectbox("selection", ['cyclic', 'random'], index=0)
        random_state = col1.slider('random_state', min_value=1, max_value=100, value=42, step=1)

        regr = MultiTaskElasticNet(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            warm_start=warm_start,
            max_iter=max_iter,
            selection=selection,
            random_state=random_state
        )

    elif ml_algorithm == "Multi Task Elastic Net CV":
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        copy_X = col1.selectbox("copy_X", [True, False], index=0)
        max_iter = col1.slider("max_iter", min_value=1000, max_value=10000, value=1000, step=100)
        selection = col1.selectbox("selection", ['cyclic', 'random'], index=0)
        random_state = col1.slider('random_state', min_value=1, max_value=100, value=42, step=1)

        regr = MultiTaskElasticNetCV(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter,
            selection=selection,
            random_state=random_state
        )

    elif ml_algorithm == "Multi Task Lasso":
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        copy_X = col1.selectbox("copy_X", [True, False], index=0)
        max_iter = col1.slider("max_iter", min_value=1000, max_value=10000, value=1000, step=100)
        selection = col1.selectbox("selection", ['cyclic', 'random'], index=0)
        random_state = col1.slider('random_state', min_value=1, max_value=100, value=42, step=1)

        regr = MultiTaskLasso(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter,
            selection=selection,
            random_state=random_state
        )

    elif ml_algorithm == "Multi Task Lasso CV":
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        copy_X = col1.selectbox("copy_X", [True, False], index=0)
        max_iter = col1.slider("max_iter", min_value=1000, max_value=10000, value=1000, step=100)
        selection = col1.selectbox("selection", ['cyclic', 'random'], index=0)
        random_state = col1.slider('random_state', min_value=1, max_value=100, value=42, step=1)

        regr = MultiTaskLassoCV(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter,
            selection=selection,
            random_state=random_state
        )

    elif ml_algorithm == "NuSVR":
        kernel = col1.selectbox("kernel", ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], index=2)
        gamma = col1.selectbox("gamma", ['scale', 'auto'], index=0)
        nu = col1.slider("nu", min_value=0.1, max_value=10.0, value=0.5, step=0.1)
        shrinking = col1.selectbox("shrinking", [True, False], index=0)

        regr = NuSVR(
            kernel=kernel,
            gamma=gamma,
            nu=nu,
            shrinking=shrinking
        )

    elif ml_algorithm == "Orthogonal Matching Pursuit":
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)

        regr = OrthogonalMatchingPursuit(
            fit_intercept=fit_intercept
        )

    elif ml_algorithm == "Orthogonal Matching Pursuit CV":
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        copy = col1.selectbox("copy", [True, False], index=0)
        regr = OrthogonalMatchingPursuitCV(
            fit_intercept=fit_intercept,
            copy=copy
        )

    elif ml_algorithm == "PLS Canonical":
        copy = col1.selectbox("copy", [True, False], index=0)
        scale = col1.selectbox("scale", [True, False], index=0)
        algorithm = col1.selectbox("algorithm", ['nipals', 'svd'], index=0)
        max_iter = col1.slider("max_iter", min_value=100, max_value=10000, value=500, step=100)
        regr = PLSCanonical(
            copy=copy,
            scale=scale,
            algorithm=algorithm,
            max_iter=max_iter
        )

    elif ml_algorithm == "PLS Regression":
        copy = col1.selectbox("copy", [True, False], index=0)
        scale = col1.selectbox("scale", [True, False], index=0)
        max_iter = col1.slider("max_iter", min_value=100, max_value=10000, value=500, step=100)
        regr = PLSRegression(
            copy=copy,
            scale=scale,
            max_iter=max_iter
        )

    elif ml_algorithm == "Passive Aggressive Regressor":
        c = col1.slider("C", min_value=1.0, max_value=100.0, value=1.0, step=0.1)
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        shuffle = col1.selectbox("shuffle", [True, False], index=0)
        warm_start = col1.selectbox("warm_start", [True, False], index=1)
        max_iter = col1.slider("max_iter", min_value=100, max_value=10000, value=1000, step=100)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)

        regr = PassiveAggressiveRegressor(
            C=c,
            fit_intercept=fit_intercept,
            shuffle=shuffle,
            warm_start=warm_start,
            max_iter=max_iter,
            random_state=random_state
        )

    elif ml_algorithm == "Poisson Regressor":
        alpha = col1.slider("alpha", min_value=1.0, max_value=100.0, value=1.0, step=0.1)
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        solver = col1.selectbox("solver", ['lbfgs', 'newton-cholesky'], index=0)
        warm_start = col1.selectbox("warm_start", [True, False], index=1)
        max_iter = col1.slider("max_iter", min_value=100, max_value=1000, value=100, step=10)

        regr = PoissonRegressor(
            alpha=alpha,
            solver=solver,
            fit_intercept=fit_intercept,
            warm_start=warm_start,
            max_iter=max_iter,
        )

    elif ml_algorithm == "Quantile Regressor":
        alpha = col1.slider("alpha", min_value=1.0, max_value=100.0, value=1.0, step=0.1)
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        solver = col1.selectbox("solver", ['highs-ds', 'highs-ipm', 'highs', 'interior-point', 'revised simplex'], index=2)

        regr = QuantileRegressor(
            alpha=alpha,
            solver=solver,
            fit_intercept=fit_intercept
        )
    elif ml_algorithm == "RANSAC Regressor":
        max_trials = col1.slider("max_trials", min_value=100, max_value=1000, value=100, step=100)
        random_state = col1.slider("random_state",  min_value=1, max_value=100, value=42, step=1)

        regr = RANSACRegressor(
            max_trials=max_trials,
            random_state=random_state
        )

    elif ml_algorithm == "Radius Neighbors Regressor":
        radius = col1.slider("radius", min_value=1.0, max_value=10.0, value=1.0, step=0.1)
        weights = col1.selectbox("weights", ["uniform", "distance"], index=0)
        algorithm = col1.selectbox("algorithm", ["auto", "ball_tree", "kd_tree", "brute"], index=0)
        leaf_size = col1.slider("leaf_size", min_value=1, max_value=100, value=30, step=1)

        regr = RadiusNeighborsRegressor(
            radius=radius,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size
        )

    elif ml_algorithm == "Random Forest Regressor":
        n_estimators = col1.slider("n_estimators", min_value=100, max_value=1000, value=100, step=10)
        criterion = col1.selectbox("criterion", ["squared_error", "absolute_error", "friedman_mse", "poisson"], index=0)
        bootstrap = col1.selectbox("bootstrap", [True, False], index=0)
        warm_start = col1.selectbox("warm_start", [True, False], index=1)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)

        regr = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion=criterion,
            bootstrap=bootstrap,
            warm_start=warm_start,
            random_state=random_state
        )

    elif ml_algorithm == "Regressor Chain":
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)
        regr = RegressorChain(
            random_state=random_state
        )

    elif ml_algorithm == "Ridge":
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        copy_X = col1.selectbox("copy_X", [True, False], index=0)
        positive = col1.selectbox("positive", [True, False], index=1)
        solver = col1.selectbox("solver", ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'], index=0)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)
        regr = Ridge(
            fit_intercept=fit_intercept,
            random_state=random_state,
            copy_X=copy_X,
            positive=positive,
            solver=solver
        )

    elif ml_algorithm == "Ridge CV":
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        regr = RidgeCV(
            fit_intercept=fit_intercept,
        )

    elif ml_algorithm == "SGD Regressor":
        loss = col1.selectbox("loss", ["l2", "l1", "elasticnet", None], index=0)
        alpha = col1.slider("alpha", min_value=0.0001, max_value=1.000, value=0.0001, step=0.0001)
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        max_iter = col1.slider("max_iter", min_value=1000, max_value=10000, value=100, step=100)

        regr = SGDRegressor(
            fit_intercept=fit_intercept,
            loss=loss,
            alpha=alpha,
            max_iter=max_iter
        )

    elif ml_algorithm == "SVR":
        kernel = col1.selectbox("kernel", ["linear", "poly", "rbf", "sigmoid", "precomputed"], index=2)
        degree = col1.slider("degree", min_value=1, max_value=100, value=3, step=1)
        epsilon = col1.slider("epsilon", min_value=0.1, max_value=100.0, value=0.1, step=0.1)
        shrinking = col1.selectbox("shrinking", [True, False], index=0)
        max_iter = col1.slider("max_iter", min_value=1000, max_value=10000, value=100, step=100)

        regr = SVR(
            kernel=kernel,
            degree=degree,
            epsilon=epsilon,
            shrinking=shrinking,
            max_iter=max_iter
        )

    elif ml_algorithm == "Theil Sen Regressor":
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        copy_X = col1.selectbox("copy_X", [True, False], index=0)
        max_iter = col1.slider("max_iter", min_value=100, max_value=10000, value=100, step=100)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)

        regr = TheilSenRegressor(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            random_state=random_state,
            max_iter=max_iter
        )

    elif ml_algorithm == "Transformed Target Regressor":
        check_inverse = col1.selectbox("check_inverse", [True, False], index=0)

        regr = TransformedTargetRegressor(
            check_inverse=check_inverse
        )

    elif ml_algorithm == "TweedieRegressor":
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        link = col1.selectbox("link", ["auto", "identity", "log"], index=0)
        solver = col1.selectbox("solver", ["lbfgs", "newton-cholesky"], index=0)
        warm_start = col1.selectbox("warm_start", [True, False], index=1)
        max_iter = col1.slider("max_iter", min_value=100, max_value=10000, value=100, step=100)

        regr = TweedieRegressor(
            fit_intercept=fit_intercept,
            link=link,
            solver=solver,
            warm_start=warm_start,
            max_iter=max_iter
        )

    if X is not None and y is not None and algo_type == "Classifiers":

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf.fit(X_train, y_train)
        model_buffer = io.BytesIO()
        joblib.dump(clf, model_buffer)
        model_buffer.seek(0)

        deployment_zip_buffer = create_zip(model_buffer=model_buffer)
        y_pred = clf.predict(X_test)


        col2.markdown("<center><h3>Metrics</h3></center>", unsafe_allow_html=True)
        col2.download_button(
            label="Download Model",
            data = model_buffer,
            file_name="model.joblib",
            mime="application/octet-stream",
            use_container_width=True
        )
        col2.download_button(
            label="Download Deployment Zip",
            data=deployment_zip_buffer,
            file_name="deployment.zip",
            mime="application/zip",
            use_container_width=True
        )
        col2.markdown(f"<b>Accuracy:</b> {accuracy_score(y_test, y_pred)}", unsafe_allow_html=True)
        col2.markdown(f"<b>Precision - Micro:</b> {precision_score(y_test, y_pred, average='micro')}", unsafe_allow_html=True)
        col2.markdown(f"<b>Recall - Micro: </b> {recall_score(y_test, y_pred, average='micro')}", unsafe_allow_html=True)
        col2.markdown(f"<b>F1 Score - Micro: </b> {f1_score(y_test, y_pred, average='micro')}", unsafe_allow_html=True)
        col2.markdown(f"<b>Precision - Macro:</b> {precision_score(y_test, y_pred, average='macro')}", unsafe_allow_html=True)
        col2.markdown(f"<b>Recall - Macro:</b> {recall_score(y_test, y_pred, average='macro')}", unsafe_allow_html=True)
        col2.markdown(f"<b>F1 Score - Macro:</b> {f1_score(y_test, y_pred, average='macro')}", unsafe_allow_html=True)
        col2.markdown(f"<b>Precision - Weighted: </b> {precision_score(y_test, y_pred, average='weighted')}", unsafe_allow_html=True)
        col2.markdown(f"<b>Recall - Weighted:</b> {recall_score(y_test, y_pred, average='weighted')}", unsafe_allow_html=True)
        col2.markdown(f"<b>F1 Score - Weighted: </b> {f1_score(y_test, y_pred, average='weighted')}", unsafe_allow_html=True)
        col2.markdown(f"<b>Classification Report:</b> {classification_report(y_test, y_pred)}", unsafe_allow_html=True)
        #col2.text(f"ROC AUC Score - OVR: {roc_auc_score(y_test, y_pred, multi_class='ovr')}")
        #col2.text(f"ROC AUC Score - OVO: {roc_auc_score(y_test, y_pred, multi_class='ovo')}")
        col2.markdown(f"<b>Confusion Matrix:</b> {confusion_matrix(y_test, y_pred)}", unsafe_allow_html=True)
        col2.markdown(f"<b>Hamming Loss: </b> {hamming_loss(y_test, y_pred)}", unsafe_allow_html=True)
        col2.markdown(f"<b>Jaccard Similarity Score: </b> {jaccard_score(y_test, y_pred, average='weighted')}", unsafe_allow_html=True)
        #col2.text(f"Log Loss: {log_loss(y_test, y_pred)}")
        col2.markdown(f"<b>Matthews Correlation Coefficient: </b> {matthews_corrcoef(y_test, y_pred)}", unsafe_allow_html=True)
        col2.markdown(f"<b>Balanced Accuracy:</b> {balanced_accuracy_score(y_test, y_pred)}", unsafe_allow_html=True)
        #col2.text(f"Precision-Recall Curve: {precision_recall_curve(y_test, y_pred)}")
        #col2.text(f"ROC Curve: {roc_curve(y_test, y_pred)}")
        col2.markdown(f"<b>Zero-One Loss:</b> {zero_one_loss(y_test, y_pred)}", unsafe_allow_html=True)


    elif X is not None and y is not None and algo_type == "Regressors":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        regr.fit(X_train_scaled, y_train)
        model_buffer = io.BytesIO()
        joblib.dump(regr, model_buffer)
        model_buffer.seek(0)

        deployment_zip_buffer = create_zip(model_buffer=model_buffer)
        y_pred = regr.predict(X_test_scaled)

        col2.markdown("<center><h3>Metrics</h3></center>", unsafe_allow_html=True)
        col2.download_button(
            label="Download Model",
            data=model_buffer,
            file_name="model.joblib",
            mime="application/octet-stream",
            use_container_width=True
        )

        col2.download_button(
            label="Download Deployment Zip",
            data=deployment_zip_buffer,
            file_name="deployment.zip",
            mime="application/zip",
            use_container_width=True
        )
        col2.markdown(f"<b>Mean Absolute Error:</b> {mean_absolute_error(y_test, y_pred)}", unsafe_allow_html=True)
        col2.markdown(f"<b>Mean Squared Error:</b> {mean_squared_error(y_test, y_pred)}",
                      unsafe_allow_html=True)
        col2.markdown(f"<b>Mean Squared Error (Squared = False): </b> {mean_squared_error(y_test, y_pred, squared=False)}", unsafe_allow_html=True)
        col2.markdown(f"<b>Mean Squared Log Error: </b> {mean_squared_log_error(y_test, y_pred)}", unsafe_allow_html=True)
        col2.markdown(f"<b>Median Absolute Error:</b> {median_absolute_error(y_test, y_pred)}",
                      unsafe_allow_html=True)
        col2.markdown(f"<b>R2 Score:</b> {r2_score(y_test, y_pred)}",
                      unsafe_allow_html=True)
        col2.markdown(f"<b>Explained Variance Score:</b> {explained_variance_score(y_test, y_pred)}",
                      unsafe_allow_html=True)


if __name__ == "__main__":
    main()

