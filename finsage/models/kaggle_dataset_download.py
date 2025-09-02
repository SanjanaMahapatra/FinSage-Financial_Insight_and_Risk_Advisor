'''
This file is for downloading the kaggle datasets using kaggle api
'''

import kaggle

kaggle.api.authenticate()

kaggle.api.dataset_download_files('computingvictor/transactions-fraud-datasets', path='finsage/data/raw/transaction_kaggle_dataset', unzip=True)