import logging
import os
from urllib.request import urlopen

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

DATA_DIR = "data"


def download_file(file_url, download_path):
    # extract file name from url
    file_name = os.path.basename(file_url)

    # Save the file in the data folder
    file_path = os.path.join(download_path, file_name)

    # check if an existing file has correct size
    if is_file_complete(file_path, url):
        logger.info(f"{file_name} already exists and has the right file size.")
        return
    else:
        logger.info(f"downloading {file_name}")

    # Ensure the data directory exists
    os.makedirs(download_path, exist_ok=True)

    download_data(file_path, file_url)


def is_file_complete(file_path, file_url):
    if os.path.exists(file_path):
        with urlopen(file_url) as response:
            try:
                total_size = int(response.info().get('Content-Length').strip())
                if os.path.getsize(file_path) == total_size:
                    return True
                else:
                    logger.info(f"{file_path} is corrupted.")
                    return False
            except Exception as e:
                logger.warning(f"unable to get file size for {file_url}: {e}")
                return True
    else:
        return False


def download_data(file_path, file_url):
    # Streaming, so we can iterate over the response.
    response = requests.get(file_url, stream=True)
    # Sizes in bytes.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(file_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Could not download file")


urls = []

############## used for context extraction ####################
# English1000 is a 985-dimensional word
# embedding feature space based on word co-occurrence in English text
# see more at https://github.com/HuthLab/deep-fMRI-dataset/
english1000_encodings = "https://github.com/HuthLab/deep-fMRI-dataset/raw/refs/heads/master/em_data/english1000sm.hf5"
urls.append(english1000_encodings)

############### used for building correlation matrices ########
# choose subjects to download fmri data for
# fMRI data for the six subjects in the experiment
# in arrays of (time x voxels) for each data collection run
# (10 stories of training data, 1 story repeated 2 times as validation data).
# https://gin.g-node.org/denizenslab/narratives_reading_listening_fmri
subjects = [
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09"
]
modalities = [
    "reading",
    "listening"
]
for subject in subjects:
    urls.append(
        f"https://gin.g-node.org/denizenslab/narratives_reading_listening_fmri/raw/master/mappers/subject{subject}_mappers.hdf")
    for modality in modalities:
        for data_type in ["trn", "val"]:
            url = f"https://gin.g-node.org/denizenslab/narratives_reading_listening_fmri/raw/master/responses/subject{subject}_{modality}_fmri_data_{data_type}.hdf"
            urls.append(url)

######### used for calculating residuals and joint models ####################
features_matrix_url = "https://gin.g-node.org/gallantlab/story_listening/raw/master/features/features_matrix.hdf"
# contain information on low level features
features_trn_new_url = "https://gin.g-node.org/denizenslab/narratives_reading_listening_fmri/raw/18fd91d109305acea443610303aa8ac992d926bb/features/features_trn_NEW.hdf"
features_val_new_url = "https://gin.g-node.org/denizenslab/narratives_reading_listening_fmri/raw/18fd91d109305acea443610303aa8ac992d926bb/features/features_val_NEW.hdf"

for url in [features_matrix_url, features_trn_new_url, features_val_new_url]:
    urls.append(url)

if __name__ == "__main__":
    for url in urls:
        logger.info(f"Downloading data from {url}")
        try:
            download_file(url, DATA_DIR)
        except Exception as e:
            logger.error(f"unable to download file from {url}: {e}")
