import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_and_clean_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Naye DAIGT V4 dataset ke hisaab se update kiya gaya logic:
    if 'generated' in df.columns:
        labels = df['generated'].values
    elif 'label' in df.columns:
        labels = df['label'].values
    else:
        raise ValueError("Error: Dataset mein 'generated' ya 'label' column nahi mila!")
        
    texts = df['text'].values
    
    print(f"Total records loaded: {len(texts)}")
    return texts, labels

def prepare_sequences(texts, max_words=10000, max_len=100):
    print("Tokenizing and padding sequences...")
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    
    # Text ko sequence of numbers mein badalna
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Neural network ke liye uniform input size ensure karna
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    
    print(f"Shape of padded sequences: {padded_sequences.shape}")
    return padded_sequences, tokenizer