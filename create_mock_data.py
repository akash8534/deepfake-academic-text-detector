import pandas as pd
import os

def generate_mock_data():
    os.makedirs('data', exist_ok=True)
    
    # 0 = Human, 1 = AI (ChatGPT/Claude/Gemini style)
    mock_texts = [
        "The utilization of multi-scale convolutional neural networks allows for robust feature extraction in latent spaces.", # AI
        "I think that neural networks are really cool because they learn from examples just like humans do.", # Human
        "In conclusion, the proposed methodology achieves state-of-the-art results across various benchmark datasets.", # AI
        "So basically we just used a simple model and it worked pretty well for our school project.", # Human
        "Furthermore, the embeddings demonstrate significant clustering when subjected to dimensionality reduction.", # AI
        "This paper is about how we can detect fake text using deep learning and python." # Human
    ]
    
    labels = [1, 0, 1, 0, 1, 0]
    
    # Data ko multiply karke 600 rows banayenge taaki neural network train ho sake
    data = {
        'text': mock_texts * 100,
        'generated': labels * 100
    }
    
    df = pd.DataFrame(data)
    df.to_csv('data/train.csv', index=False)
    print("Success! Mock dataset created at 'data/train.csv'")

if __name__ == "__main__":
    generate_mock_data()