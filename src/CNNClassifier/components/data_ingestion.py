import os
import ssl
import shutil
import urllib.request as request
import zipfile
from pathlib import Path

from CNNClassifier import logger
from CNNClassifier.utils.common import get_size
from CNNClassifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        local_file = Path(self.config.local_data_file)

        if not os.path.exists(local_file):
            local_file.parent.mkdir(parents=True, exist_ok=True)

            context = ssl._create_unverified_context()
            with request.urlopen(self.config.source_URL, context=context) as response:
                with open(local_file, "wb") as f:
                    shutil.copyfileobj(response, f)

            logger.info(f"Downloaded file to {local_file}")
        else:
            logger.info(f"File already exists of size: {get_size(local_file)}")

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)