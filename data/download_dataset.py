import requests
import zipfile
import os
from tqdm import tqdm


def download_and_unzip(cf):

    # config
    seafile_url = cf.seafile_url
    path_data_dir = cf.path_data_dir

    # Create directory
    os.makedirs(path_data_dir, exist_ok=True)

    # Check if data exists
    path_data_mvtec = cf.path_data_mvtec

    if os.path.exists(path_data_mvtec):
        print(f"Data already exists in: {path_data_dir}")
        print(f"Dataset Image MVTec: {path_data_mvtec}")
        return

    print(f"Downloading from Seafile... {seafile_url}")
    zip_path = os.path.join(path_data_dir, "temp_download.zip")

    # Download file (302 redirect will be followed automatically)
    response = requests.get(seafile_url, allow_redirects=True, stream=True)

    # Save file
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded: {os.path.getsize(zip_path) / (1024**2):.2f} MB")

    # Extract
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        files = [
            f for f in zip_ref.namelist() if not ("__MACOSX/" in f or f.startswith("."))
        ]
        for file in tqdm(files, desc="Extracting"):
            zip_ref.extract(file, path_data_dir)

    os.remove(zip_path)
    print(f"Complete! Files in: {path_data_dir}")
