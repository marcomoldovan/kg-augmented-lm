import requests
import os


def download_wikidata_dataset(destination_dir="../data/wikidata5m/"):
    # URL for the dataset
    urls = {
        "wikidata5m_alias.tar.gz": "https://www.dropbox.com/s/lnbhc8yuhit4wm5/wikidata5m_alias.tar.gz?dl=1",
        "wikidata5m_text.txt.gz": "https://www.dropbox.com/s/7jp4ib8zo3i6m10/wikidata5m_text.txt.gz?dl=1",
        "wikidata5m_all_triplet.txt.gz": "https://www.dropbox.com/s/563omb11cxaqr83/wikidata5m_all_triplet.txt.gz?dl=1",
        "wikidata5m_inductive.tar.gz": "https://www.dropbox.com/s/csed3cgal3m7rzo/wikidata5m_inductive.tar.gz?dl=1",
        "wikidata5m_transductive.tar.gz": "https://www.dropbox.com/s/6sbhm0rwo4l73jq/wikidata5m_transductive.tar.gz?dl=1"
    }


    # Make sure the destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for k, v in urls.items():

        # Path to save the file
        save_path = os.path.join(destination_dir, k)
        
        if not os.path.exists(save_path):        
            # Download the file
            response = requests.get(v, stream=True)
            response.raise_for_status()  # Raise an error for failed requests
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)