"""
Download and extract the satellite land classification dataset.

Source: IBM Cloud Object Storage (used in the IBM AI Capstone course dataset).
"""

import os
import tarfile
import requests

DATA_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5vTzHBmQUaRNJQe5szCyKw/images-dataSAT.tar"
RAW_DATA_DIR = os.path.join("data", "raw")
ARCHIVE_PATH = os.path.join(RAW_DATA_DIR, "images-dataSAT.tar")


def download_file(url: str, destination: str) -> None:
    """Download a file from a URL to a local destination, streaming to avoid high memory use."""
    if os.path.exists(destination):
        print(f"File already exists, skipping download: {destination}")
        return

    print(f"Downloading dataset from:\n{url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size:
                percent = downloaded / total_size * 100
                print(f"\rProgress: {percent:.1f}%", end="")

    print("\nDownload complete.")


def extract_archive(archive_path: str, extract_to: str) -> None:
    """Extract a .tar archive into the target directory."""
    print(f"Extracting {archive_path} to {extract_to} ...")
    with tarfile.open(archive_path, "r") as tar:
        tar.extractall(path=extract_to)
    print("Extraction complete.")


def main() -> None:
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    download_file(DATA_URL, ARCHIVE_PATH)
    extract_archive(ARCHIVE_PATH, RAW_DATA_DIR)


if __name__ == "__main__":
    main()