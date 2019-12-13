gsutil cp gs://cs221_chexpert/dev/v1/train_set.csv data/train.csv
gsutil cp gs://cs221_chexpert/dev/v1/valid_set.csv data/valid.csv
python data/data_processing.py
