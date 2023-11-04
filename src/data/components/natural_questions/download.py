import requests
import os

def download_nq_dataset(destination_dir="../data/natural_questions/"):
    # URL for the dataset
    url = "https://storage.googleapis.com/natural_questions/v1.0-simplified/simplified-nq-train.jsonl.gz"
    
    # Make sure the destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Path to save the file
    save_path = os.path.join(destination_dir, "simplified-nq-train.jsonl.gz")
    
    # Download the file
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error for failed requests
    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    print(f"Downloaded the dataset to {save_path}")
