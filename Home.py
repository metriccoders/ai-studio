import streamlit as st
from streamlit_lottie import st_lottie
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, confusion_matrix, hamming_loss, jaccard_score, log_loss, matthews_corrcoef, balanced_accuracy_score, precision_recall_curve, roc_curve, zero_one_loss
from sklearn.model_selection import train_test_split


st.markdown("<center><h1>Artificial Intelligence Studio</h1></center>", unsafe_allow_html=True)
#st.lottie("https://lottie.host/f9ecc8cd-9a0e-49f5-bfbe-89bb59ca794b/Qnv20SfUVi.json", height=50, width=50, quality="high")

st.markdown("<center><h4><b>By Metric Coders</b></h4></center>", unsafe_allow_html=True)

dataset = load_iris()
clf = AdaBoostClassifier
ml_algorithm = st.selectbox("ML Algorithm", ["AdaBoost Classifier",
                                             "Bagging Classifier",
                                             "BernoulliNB",
                                             "Calibrated Classifier CV",
                                             "Decision TreeClassifier",
                                             "Extra Trees Classifier",
                                             "Gradient Boosting Classifier",
                                             "KNeighbors Classifier",
                                             "Random Forest Classifier"
                                             ], index=0)

dataset_option = st.selectbox("Dataset", ["Iris", "Digits", "Wine", "Breast Cancer"], index=0)


X = dataset.data
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

if dataset_option == "Iris":
    dataset = load_iris()
elif dataset_option == "Digits":
    dataset = load_digits()
elif dataset_option == "Wine":
    dataset = load_wine()
elif dataset_option == "Breast Cancer":
    dataset = load_breast_cancer()


X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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
    warm_start = col1.selectbox("warm_start", [True, False], index=0)
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




clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)





col2.markdown("<center><h3>Metrics</h3></center>", unsafe_allow_html=True)
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








