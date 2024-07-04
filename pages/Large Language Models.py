import streamlit as st
import io
import sys

import re
from sklearn.model_selection import train_test_split
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

st.set_page_config(layout="wide", initial_sidebar_state="expanded")
#
#
# # Load a default dataset (e.g., IMDb reviews)
# dataset = load_dataset('imdb')
#
# # Initialize the tokenizer and model
# model_name = 'bert-base-uncased'
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
#
#
# def tokenize_function(examples):
#     return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128, return_tensors="tf")
#
# tokenized_datasets = dataset.map(tokenize_function, batched=True)
#
# # Convert tokenized datasets to tf.data.Dataset
# def to_tf_dataset(tokenized_dataset, shuffle=False, batch_size=8):
#     dataset = tf.data.Dataset.from_tensor_slices((
#         {k: tokenized_dataset[k] for k in tokenizer.model_input_names},
#         tokenized_dataset["label"]
#     ))
#     if shuffle:
#         dataset = dataset.shuffle(buffer_size=len(tokenized_dataset))
#     dataset = dataset.batch(batch_size)
#     return dataset
#
# def build_text_files(data_text, dest_path):
#     f = open(dest_path, 'w')
#     data = ''
#     for texts in data_text:
#         summary = str(texts).strip()
#         summary = re.sub(r"\s", " ", summary)
#         data += summary + "  "
#     f.write(data)
#


def main():
    st.markdown("<center><h1>Artificial Intelligence (AI) Studio</h1></center>", unsafe_allow_html=True)
    #st.lottie("https://lottie.host/f9ecc8cd-9a0e-49f5-bfbe-89bb59ca794b/Qnv20SfUVi.json", height=50, width=50, quality="high")
    st.markdown("<center><h4><b>By Metric Coders</b></h4></center>", unsafe_allow_html=True)
    st.markdown("<center><h4><b>A No-Code Platform to train and deploy your Large Language Models</b></h4></center>",
                unsafe_allow_html=True)

    st.markdown("<center><h4><b>Fine-tune your custom LLMs</b></h4></center>", unsafe_allow_html=True)
    st.markdown("<center><h5><b>Fine-tuning LLM disabled due to lack of processing power</b></h5></center>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Training Data in a CSV file", disabled=True, type=["txt"])

    num_of_epochs = st.slider("num_epochs", min_value=1, max_value=2, value=1, step=1)

    batch_size = st.slider("batch_size", min_value=2, max_value=4, value=2, step=1)

    st.button("Fine-Tune", disabled=True, use_container_width=True)

    # if uploaded_file is not None:
    #     content = uploaded_file.getvalue().decode("utf-8")
    #     st.write("Preview of Dataset")
    #     text = content.splitlines()
    #     train, test = train_test_split(text, test_size=0.15)
    #
    #     build_text_files(train, 'train_dataset.txt')
    #     build_text_files(test, 'test_dataset.txt')
    #
    #     # Load a default dataset (e.g., IMDb reviews)
    #     dataset = load_dataset('imdb')
    #
    #     model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    #
    #     train_dataset = to_tf_dataset(tokenized_datasets['train'], shuffle=True, batch_size=batch_size)
    #     eval_dataset = to_tf_dataset(tokenized_datasets['test'], batch_size=batch_size)
    #
    #     # Compile the model
    #     optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=2e-5)
    #
    #     model.compile(optimizer=optimizer, metrics=['accuracy'])
    #
    #     # Train the model
    #     model.fit(train_dataset, validation_data=eval_dataset, epochs=num_of_epochs)


if __name__ == "__main__":
    main()