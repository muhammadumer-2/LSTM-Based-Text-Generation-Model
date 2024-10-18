# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
import PyPDF2


# Path to the PDF file
pdf_file_path = '/content/DS.pdf'

# Open the PDF file
with open(pdf_file_path, 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    number_of_pages = len(reader.pages)

    # Read each page
    for page in range(number_of_pages):
        text = reader.pages[page].extract_text()
        print(f'Page {page + 1}:\n{text}\n')

# Split the text into sentences
sentences = sent_tokenize(text)
len(sentences)

# Create a DataFrame with the text data
data = {'text': sentences}
df = pd.DataFrame(data)

# Task 2: Data Preprocessing
# Clean and preprocess the text data
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))
    return text

# Apply preprocessing to the text data
df['clean_text'] = df['text'].apply(preprocess_text)

# Tokenize the text into sequences of integers
tokenizer = Tokenizer()

tokenizer.fit_on_texts(df['clean_text'])

total_words = len(tokenizer.word_index) + 1

# Convert text to sequences of integers
input_sequences = []
for line in df['clean_text']:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences to ensure uniform length
max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Create predictors and labels
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# Task 3: Model Architecture
# Design and implement the LSTM model
model = Sequential()

model.add(Embedding(total_words, 100))

model.add(LSTM(150))

model.add(Dense(total_words, activation='softmax'))

# Task 4: Model Training
# Compile and train the LSTM model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=1)

# Task 5: Evaluation
# Evaluate the model on a validation dataset
# For simplicity, let's use a train-test split on the provided data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

_, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)

# Task 6: Creativity and Language Analysis
# Generate text using the trained model
def generate_text(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list)[0]
        predicted_index = np.argmax(predicted_probs)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text


# Generate text using the trained model
generated_text = generate_text("The Gaussian distribution", next_words=15)
print("Generated Text:", generated_text)

# Perform more linguistic analysis as needed