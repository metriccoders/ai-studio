import streamlit as st
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

st.title("AI Studio by Metric Coders")


dataset = load_iris()
X = dataset.data
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

col1, col2 = st.columns(2)

criterion = col1.selectbox("criterion", ["gini", "entropy", "log_loss"], placeholder="Choose a criterion", index=0)
splitter= col1.selectbox("splitter", ["best", "random"], index=0)
max_depth = col1.slider("max_depth", min_value=3, max_value=200, value=3, step=1)
min_samples_split = col1.slider("min_samples_split", min_value=2, max_value=100, value=2, step=1)
min_samples_leaf = col1.slider("min_samples_leaf", min_value=1, max_value=200, value=1, step=1)
min_weight_fraction_leaf = col1.slider("min_weight_fraction_leaf", min_value=0.0, max_value=0.5, value=0.0, step=0.1)
max_features = col1.selectbox("max_features", ["sqrt", "log2", None], placeholder="Choose max. number of features",
                             index=2)
ccp_alpha=col1.slider("ccp_alpha", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
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

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

col2.header(f"Accuracy: {accuracy_score(y_test, y_pred)}")








