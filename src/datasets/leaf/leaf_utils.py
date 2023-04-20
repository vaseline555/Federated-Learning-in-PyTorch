import os
import zipfile
import logging
import requests
import torchvision

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ['letter_to_vec', 'word_to_indices', 'download_data']



ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
URL = {
    'femnist': [
        'https://s3.amazonaws.com/nist-srd/SD19/by_class.zip',
        'https://s3.amazonaws.com/nist-srd/SD19/by_write.zip'
    ],
    'shakespeare': ['http://www.gutenberg.org/files/100/old/1994-01-100.zip'],
    'sent140': [
        'http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip',
        'http://nlp.stanford.edu/data/glove.6B.zip' # GloVe embedding for vocabularies
    ],
    'celeba': [ 
        '1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS', # Google Drive link ID
        '0B7EVK8r0v71pblRyaVFSWGxPY0U', # Google Drive link ID
        'https://cseweb.ucsd.edu/~weijian/static/datasets/celeba/img_align_celeba.zip' # img_align_celeba.zip
    ],
    'reddit': ['1ISzp69JmaIJqBpQCX-JJ8-kVyUns8M7o']  # Google Drive link ID
}
OPT = { # md5 checksum if direct URL link is provided, file name if Google Drive link ID is provided  
    'femnist': ['79572b1694a8506f2b722c7be54130c4', 'a29f21babf83db0bb28a2f77b2b456cb'],
    'shakespeare': ['b8d60664a90939fa7b5d9f4dd064a1d5'],
    'sent140': ['1647eb110dd2492512e27b9a70d5d1bc', '056ea991adb4740ac6bf1b6d9b50408b'],
    'celeba': ['identity_CelebA.txt', 'list_attr_celeba.txt', '00d2c5bc6d35e252742224ab0c1e8fcb'],
    'reddit': ['reddit_subsampled.zip']
}
    
    
def letter_to_vec(letter):
    """Convert given letter into index.
    """
    index = ALL_LETTERS.find(letter)
    return index

def word_to_indices(word):
    """Convert a word into a list of character indices.
    """
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices

def download_data(download_root, dataset_name):
    """Download data from Google Drive and extract if it is archived.
    """
    def _get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None
        
    def _save_response_content(download_root, response):
        CHUNK_SIZE = 32768
        with open(download_root, 'wb') as file:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keezp-alive new chunks
                    file.write(chunk)
                        
    def _download_file_from_google_drive(download_root, file_name, identifier):
        BASE_URL = 'https://docs.google.com/uc?export=download'
            
        session = requests.Session()
        response = session.get(BASE_URL, params={'id': identifier, 'confirm': 1}, stream=True)
        token = _get_confirm_token(response)

        if token:
            params = {'id': identifier, 'confirm': token }
            response = session.get(BASE_URL, params=params, stream=True)
        _save_response_content(os.path.join(download_root, file_name), response)    
        print(f'...successfully downloaded file `{file_name}` at `{download_root}`!')
            
        if '.zip' in file_name:
            with zipfile.ZipFile(os.path.join(download_root, file_name), 'r', compression=zipfile.ZIP_STORED) as zip_file:
                zip_file.extractall(download_root)
            print(f'...successfully extracted `{file_name}` at `{download_root}`!')
                
    # download data from web
    logger.info(f'[LEAF - {dataset_name.upper()}] Start downloading data...!')
    try:
        for (url, opt) in zip(URL[dataset_name], OPT[dataset_name]):
            if 'http' not in url:
                _download_file_from_google_drive(download_root, opt, url)
            else:
                torchvision.datasets.utils.download_and_extract_archive(
                    url=url, 
                    download_root=download_root, 
                    md5=opt, 
                    remove_finished=True
                )
        else:
            logger.info(f'[LEAF - {dataset_name.upper()}] ...finished downloading data!')
    except:
        logger.exception(url)
        raise Exception(url)
