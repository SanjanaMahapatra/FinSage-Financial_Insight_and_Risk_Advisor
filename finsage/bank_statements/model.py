import kaggle


kaggle.api.authenticate()

kaggle.api.dataset_download_files('devildyno/indian-bank-statement-one-year', unzip=True, path=".")
kaggle.api.dataset_download_files('abutalhadmaniyar/bank-statements-dataset', unzip=True, path=".")
