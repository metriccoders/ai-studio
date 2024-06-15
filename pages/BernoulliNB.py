import streamlit as st
from sklearn.datasets import load_iris
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

st.title("SKLearn Studio by Metric Coders")


dataset = load_iris()
X = dataset.data
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

col1, col2 = st.columns(2)
alpha = col1.slider("alpha", min_value=1.0, max_value=10.0, value=1.0, step=0.1)
force_alpha=col1.selectbox("force_alpha", [True, False], index=0)
binarize=col1.slider("binarize", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
fit_prior=col1.selectbox("fit_prior", [True, False], index=0)

clf = BernoulliNB(
    alpha=alpha,
    force_alpha=force_alpha,
    binarize=binarize,
    fit_prior=fit_prior
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

col2.header(f"Accuracy: {accuracy_score(y_test, y_pred)}")








