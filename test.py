import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# Load a default dataset (e.g., IMDb reviews)
dataset = load_dataset('imdb')

# Initialize the tokenizer and model
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128, return_tensors="tf")

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Convert tokenized datasets to tf.data.Dataset
def to_tf_dataset(tokenized_dataset, shuffle=False, batch_size=8):
    dataset = tf.data.Dataset.from_tensor_slices((
        {k: tokenized_dataset[k] for k in tokenizer.model_input_names},
        tokenized_dataset["label"]
    ))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(tokenized_dataset))
    dataset = dataset.batch(batch_size)
    return dataset

train_dataset = to_tf_dataset(tokenized_datasets['train'], shuffle=True, batch_size=32)
eval_dataset = to_tf_dataset(tokenized_datasets['test'], batch_size=32)

# Compile the model
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=2e-5)


model.compile(optimizer=optimizer, metrics=['accuracy'])

# Train the model
model.fit(train_dataset, validation_data=eval_dataset, epochs=3)