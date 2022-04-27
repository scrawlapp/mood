from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import warnings
import nlp
import pickle
import json
import os

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
test = dataset['test']
text, labels = get_text(test)
tokenizer = Tokenizer(num_words=10000, oov_token='<UNK>')
tokenizer.fit_on_texts(text)
padded_train_sequences = get_sequences(tokenizer, text) # embedded sequence for training data
classes = set(labels)
classes_to_index = dict((c, i) for i, c in enumerate(classes))
index_to_classes = dict((v, k) for k, v in classes_to_index.items())
names_to_ids = lambda labels: np.array([classes_to_index.get(x) for x in labels])
train_labels = names_to_ids(labels)

with open('model.pickle', 'rb') as file:
    model = pickle.load(file)


class MoodServer(BaseHTTPRequestHandler):

    def getMood(self, userInput):
        message = []
        message.append(userInput)
        seq = tokenizer.texts_to_sequences(message)
        padded = pad_sequences(seq, maxlen=50, padding='post')
        p = model.predict(np.expand_dims(padded, axis=0)[0])
        return index_to_classes[np.argmax(p)]

    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, content-type")
        self.end_headers()

    def do_POST(self):
        if self.path == '/api':
            content_length = int(self.headers['Content-Length'])
            data = json.loads(self.rfile.read(content_length).decode('UTF-8'))
            result = {}
            result['message'] = self.getMood(data['userInput'])
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('UTF-8'))

server = HTTPServer(('', int(os.environ['PORT'])), MoodServer)
server.serve_forever()
