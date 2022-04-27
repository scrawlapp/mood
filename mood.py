# dependencies
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import nlp
import random
import warnings
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# get the text and labels as lists from the dataframe.
def get_text(data):             
    text = [x['text'] for x in data]
    labels = [x['label'] for x in data]
    return text, labels

# to add padding
def get_sequences(tokenizer, text):
    sequences = tokenizer.texts_to_sequences(text)
    padded_sequences = pad_sequences(sequences, truncating='post', maxlen=50, padding='post')
    return padded_sequences

# prepare data
warnings.filterwarnings('ignore')
dataset = nlp.load_dataset('emotion') 
train = dataset['train']
val = dataset['validation']
test = dataset['test']
text, labels = get_text(train)
tokenizer = Tokenizer(num_words=10000, oov_token='<UNK>')
tokenizer.fit_on_texts(text)
padded_train_sequences = get_sequences(tokenizer, text) # embedded sequence for training data
classes = set(labels)
classes_to_index = dict((c, i) for i, c in enumerate(classes))
index_to_classes = dict((v, k) for k, v in classes_to_index.items())
names_to_ids = lambda labels: np.array([classes_to_index.get(x) for x in labels])
train_labels = names_to_ids(labels)

# defining the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(10000, 16, input_length=50),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
    tf.keras.layers.Dense(6, activation='softmax')
])

# compiling the model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# getting the number of nodes and connections on all layers.
model.summary()

# preparing the validation set.
val_text, val_labels = get_text(val)
val_sequences = get_sequences(tokenizer, val_text)
val_labels = names_to_ids(val_labels)

# running the model for training.
h = model.fit(
    padded_train_sequences, train_labels,
    validation_data=(val_sequences, val_labels),
    epochs=15,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)
    ]
)
# preparing the test set.
test_text, test_labels = get_text(test)
test_sequences = get_sequences(tokenizer, test_text)
test_labels = names_to_ids(test_labels)

# evaluation of the accuracy of the model.
eval = model.evaluate(test_sequences, test_labels)

with open('model.pickle', 'wb') as file:
    pickle.dump(model, file)

# seeing random examples of the test data with actual and predicted emotion.
for a in range(0,5):
  i = random.randint(0, len(test_labels) - 1)
  print('Text:', test_text[i])
  print('Actual Emotion:', index_to_classes[test_labels[i]])
  p = model.predict(np.expand_dims(test_sequences[i], axis=0))[0]
  print('Predicted Emotion:', index_to_classes[np.argmax(p)], '\n')