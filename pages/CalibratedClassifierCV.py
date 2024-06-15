import streamlit as st
from sklearn.datasets import load_iris
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

st.title("SKLearn Studio by Metric Coders")


dataset = load_iris()
X = dataset.data
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

col1, col2 = st.columns(2)

method=col1.selectbox("method", ["sigmoid", "istonic"], index=0)
cv=col1.slider("cv", min_value=2, max_value=10, value=2, step=1)
n_jobs=col1.slider("n_jobs", min_value=1, max_value=10, value=5, step=1)
clf = CalibratedClassifierCV(
    method=method,
    cv=cv,
    n_jobs=n_jobs
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

col2.header(f"Accuracy: {accuracy_score(y_test, y_pred)}")








