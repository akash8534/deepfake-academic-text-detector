import os
# Yeh line TensorFlow ke faltu INFO aur WARNING logs ko terminal se hata degi
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

# Humare custom modules import kar rahe hain
from src.preprocess import load_and_clean_data, prepare_sequences
from src.model import build_hybrid_cnn_lstm

# Naye imports for fixing the bugs
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences

def plot_history(history):
    print("Generating training graphs...")
    plt.figure(figsize=(12, 4))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    print("=== Deepfake Academic Text Detection Pipeline ===")
    
    data_path = 'data/train.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please make sure the real dataset is placed.")
        return
        
    # 1. Data Load & Clean
    texts, labels = load_and_clean_data(data_path)
    
    # 2. Sequence Preprocessing
    max_words = 10000
    max_len = 100
    X, tokenizer = prepare_sequences(texts, max_words=max_words, max_len=max_len)
    y = labels
    
    # 3. Train-Test Split (80% training, 20% testing)
    print("Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Build Model
    model = build_hybrid_cnn_lstm(max_words=max_words, max_len=max_len)
    model.summary()
    
    # OVERFITTING FIX: Early Stopping lagana
    # Agar 2 epochs tak validation loss improve nahi hota, toh training ruk jayegi
    # aur sabse best weights restore ho jayenge.
    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    
    # 5. Train the Model
    print("\nStarting Model Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,          
        batch_size=32,
        callbacks=[early_stop], # Early stopping active
        verbose=1
    )
    
    # 6. Evaluate and Plot
    loss, accuracy, auc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal Validation Accuracy: {accuracy * 100:.2f}%")
    print(f"Final Validation AUC: {auc:.4f}")
    
    plot_history(history)
    
    # 7. Test a new sentence (Live Prediction) - BUG FIXED
    print("\n--- Live Prediction Test ---")
    test_sentences = [
        "The utilization of multi-scale convolutional neural networks allows for robust feature extraction.", # AI like
        "I really loved working on this project because it taught me a lot about python." # Human like
    ]
    
    # Yahan hum existing 'tokenizer' use kar rahe hain (naya train nahi kar rahe)
    test_seq = tokenizer.texts_to_sequences(test_sentences)
    test_padded = pad_sequences(test_seq, maxlen=max_len, padding='post', truncating='post')
    
    predictions = model.predict(test_padded)
    for i, pred in enumerate(predictions):
        label = "AI Generated" if pred[0] > 0.5 else "Human Written"
        print(f"Text: '{test_sentences[i]}'\nPrediction: {label} (AI Probability: {pred[0]*100:.1f}%)\n")

if __name__ == "__main__":
    main()