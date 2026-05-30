import kagglehub
import shutil
import os

def setup_dataset():
    print("Downloading dataset from Kaggle...")
    path = kagglehub.competition_download('llm-detect-ai-generated-text')
    
    os.makedirs('data', exist_ok=True)
    
    source_file = os.path.join(path, "train_essays.csv")
    destination_file = "data/train.csv"

    if os.path.exists(source_file):
        shutil.copy(source_file, destination_file)
        print(f"Success! Dataset moved to: {destination_file}")
    else:
        print("Error: Dataset file not found.")

if __name__ == "__main__":
    setup_dataset()