import streamlit as st
# from keras import layers
# from keras import models
# import numpy as np
# import io
# import sys
#
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
#
# if 'widgets' not in st.session_state:
#     st.session_state.widgets = []
#
#
# def add_widget():
#     widget_id = len(st.session_state.widgets)
#     st.session_state.widgets.append(widget_id)
#
# def remove_widget(widget_id):
#     st.session_state.widgets.remove(widget_id)
#
# def update_model(model):
#     for widget_id in st.session_state.widgets:
#         layer_value = st.session_state.get(f"layer_{widget_id}", None)
#         activation_function_value = st.session_state.get(f"activation_{widget_id}", None)
#         st.write(f"Layer {widget_id}: Layer Value = {layer_value}, Activation Value = {activation_function_value}")
#         model.add(layers.Dense(layer_value, ))
#
# def print_model_summary(model):
#     # Create a buffer to capture the model summary
#     buffer = io.StringIO()
#     # Redirect standard output to the buffer
#     sys.stdout = buffer
#     # Print the model summary
#     model.summary()
#     # Reset standard output
#     sys.stdout = sys.__stdout__
#     # Get the content of the buffer
#     summary_str = buffer.getvalue()
#     # Print the summary
#     return summary_str
#
#
# # Function to print model.fit output
# def print_model_fit(model, X, y, epochs=10, batch_size=32):
#     # Create a buffer to capture the model fit output
#     buffer = io.StringIO()
#     # Redirect standard output to the buffer
#     sys.stdout = buffer
#     # Train the model
#     model.fit(X, y, epochs=epochs, batch_size=batch_size)
#     # Reset standard output
#     sys.stdout = sys.__stdout__
#     # Get the content of the buffer
#     fit_output = buffer.getvalue()
#     # Print the output
#     return fit_output
#
def main():
    st.markdown("<center><h1>Artificial Intelligence (AI) Studio</h1></center>", unsafe_allow_html=True)
    #st.lottie("https://lottie.host/f9ecc8cd-9a0e-49f5-bfbe-89bb59ca794b/Qnv20SfUVi.json", height=50, width=50, quality="high")
    st.markdown("<center><h4><b>By Metric Coders</b></h4></center>", unsafe_allow_html=True)
    st.markdown("<center><h4><b>A No-Code Platform to train and deploy your Large Language Models</b></h4></center>",
             unsafe_allow_html=True)
    st.markdown("<center><h5><b>Fine-tuning LLM disabled due to lack of processing power</b></h5></center>",
                unsafe_allow_html=True)
#     num_of_samples = st.slider("Number of Samples", min_value=1000, max_value=20000, value=1000, step=100)
#     num_of_features = st.slider("Number of Features", min_value=2, max_value=20, value=5, step=1)
#     num_of_outputs = st.slider("Number of Outputs", min_value=1, max_value=num_of_features-1,value=num_of_features-1, step=1)
#
#     if st.button("Add a Layer", use_container_width=True):
#         add_widget()
#
#     for widget_id in st.session_state.widgets:
#         cols = st.columns((3, 2, 1))  # Adjust column sizes as needed
#         with cols[0]:
#             st.slider(f"Layer {widget_id}", 1, 1024, key=f"layer_{widget_id}")
#         with cols[1]:
#             st.selectbox(f"Activation Function {widget_id}",
#                          ['relu',
#                         'sigmoid',
#                         'softmax',
#                         'softplus',
#                         'softsign',
#                         'tanh',
#                         'selu',
#                         'elu',
#                         'exponential'], key=f"activation_{widget_id}")
#         with cols[2]:
#             st.write("Remove a Layer")
#             if st.button("Remove", key=f"remove_{widget_id}", use_container_width=True):
#                 remove_widget(widget_id)
#                 st.experimental_rerun()
#
#     X = np.random.rand(num_of_samples, num_of_features)
#     y = np.random.rand(num_of_samples, num_of_outputs)
#     model = models.Sequential()
#     model.add(layers.Dense(128, input_dim=num_of_features, activation="relu"))
#     update_model(model)
#     model.add(layers.Dense(num_of_outputs, activation="sigmoid"))
#     model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy", "recall", "auc", "precision"])
#
#
#    # col1, col2 = st.columns(2)
#     st.markdown("<h3>Model Details</h3>",unsafe_allow_html=True)
#     st.write(print_model_summary(model))
#
#     st.markdown("<h3>Model Results</h3>", unsafe_allow_html=True)
#     st.write(print_model_fit(model, X, y, epochs=10, batch_size=16))
#
#
#
if __name__ == "__main__":
    main()