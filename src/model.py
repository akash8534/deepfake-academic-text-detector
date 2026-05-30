import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Conv1D, LSTM, Dense, Dropout, Concatenate, GlobalMaxPooling1D
from tensorflow.keras.models import Model

def build_hybrid_cnn_lstm(max_words=10000, max_len=100, embedding_dim=128):
    print("Building Hybrid CNN-LSTM Model...")
    
    # 1. Input Layer
    input_text = Input(shape=(max_len,), dtype='int32', name="Text_Input")
    
    # 2. Latent Space Embedding (Words to Vectors)
    embedding_layer = Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len)(input_text)
    
    # 3. Multi-scale CNNs (Forensic Stylometric Pattern Extraction)
    # Branch A: Looks at 3 words at a time (Trigrams)
    cnn_branch_1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(embedding_layer)
    cnn_pool_1 = GlobalMaxPooling1D()(cnn_branch_1)
    
    # Branch B: Looks at 5 words at a time (5-grams)
    cnn_branch_2 = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(embedding_layer)
    cnn_pool_2 = GlobalMaxPooling1D()(cnn_branch_2)
    
    # Combine both CNN features
    cnn_combined = Concatenate()([cnn_pool_1, cnn_pool_2])
    cnn_combined = Dropout(0.4)(cnn_combined)
    
    # 4. LSTM Layer (Temporal Drift Analysis)
    # We pass the original embeddings to LSTM to understand the long-term sentence structure
    lstm_out = LSTM(64, return_sequences=False)(embedding_layer)
    lstm_out = Dropout(0.4)(lstm_out)
    
    # 5. Fusion Layer (Combining CNN spatial features + LSTM temporal features)
    fusion = Concatenate()([cnn_combined, lstm_out])
    dense_1 = Dense(32, activation='relu')(fusion)
    
    # 6. Classification Output (0 for Human, 1 for AI)
    output = Dense(1, activation='sigmoid', name="Classification_Output")(dense_1)
    
    model = Model(inputs=input_text, outputs=output)
    
    # Compile with Binary Crossentropy for Binary Classification
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    
    return model