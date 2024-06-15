import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

st.title("SKLearn Studio by Metric Coders")


dataset = load_iris()
X = dataset.data
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

col1, col2 = st.columns(2)

n_estimators = col1.slider("n_estimators", min_value=1, max_value=2000, value=10, step=1)
max_samples = col1.slider("max_samples", min_value=1, max_value=10, value=1, step=1)
max_features=col1.slider("max_features", min_value=1.0, max_value=5.0, value=0.1)
bootstrap = col1.selectbox("bootstrap", [True, False], index=0)
bootstrap_features = col1.selectbox("bootstrap_features", [True, False], index=0)
oob_score=col1.selectbox("oob_score", [True, False], index=0)
warm_start=col1.selectbox("warm_start", [True, False], index=0)
n_jobs=col1.slider("n_jobs", min_value=1, max_value=10, value=1, step=1)
random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)

clf = BaggingClassifier(
    n_estimators=n_estimators,
    max_samples =max_samples,
    max_features=max_features,
    bootstrap=bootstrap,
    bootstrap_features=bootstrap_features,
    oob_score=oob_score,
    warm_start=warm_start,
    n_jobs=n_jobs,
    random_state=random_state
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

col2.header(f"Accuracy: {accuracy_score(y_test, y_pred)}")








