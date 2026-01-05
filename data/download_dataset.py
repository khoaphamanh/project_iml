import requests
import zipfile
import os
from tqdm import tqdm

# Seafile share link (extract token from your URL)
seafile_url = "https://seafile.cloud.uni-hannover.de/f/f2c6136c82534a15b04c/?dl=1"


def download_and_unzip(seafile_url, extract_to="./dataset"):
    # Create directory
    os.makedirs(extract_to, exist_ok=True)

    # Check if data exists
    path_data_image = os.path.join(extract_to, "dataset_image")
    path_data_video = os.path.join(extract_to, "dataset_BDDA")

    if os.path.exists(path_data_image) or os.path.exists(path_data_video):
        print(f"Data already exists in: {extract_to}")
        print(f"Dataset Image: {path_data_image}")
        print(f"Dataset Video: {path_data_video}")
        return

    print(f"Downloading from Seafile... {seafile_url}")
    zip_path = os.path.join(extract_to, "temp_download.zip")

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
            zip_ref.extract(file, extract_to)

    os.remove(zip_path)
    print(f"Complete! Files in: {extract_to}")


# Usage
dir_curent = os.path.dirname(os.path.abspath(__file__))
download_and_unzip(
    seafile_url=seafile_url,
    extract_to=dir_curent,
)
