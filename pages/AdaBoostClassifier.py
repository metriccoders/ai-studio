import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

st.title("SKLearn Studio by Metric Coders")


dataset = load_iris()
X = dataset.data
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

col1, col2 = st.columns(2)

n_estimators = col1.slider("n_estimators", min_value=1, max_value=2000, value=50, step=1)
learning_rate = col1.slider("learning_rate",min_value=1.0, max_value=10.0, value=1.0, step=0.1)
algorithm=col1.selectbox("algorithm", ["SAMME", "SAMME.R"], index=0)
criterion = col1.selectbox("criterion", ["gini", "entropy", "log_loss"], placeholder="Choose a criterion")
random_state=col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)
clf = AdaBoostClassifier(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    algorithm=algorithm,
    random_state=random_state
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

col2.header(f"Accuracy: {accuracy_score(y_test, y_pred)}")








