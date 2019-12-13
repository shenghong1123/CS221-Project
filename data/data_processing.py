import os
import pandas as pd

train_csv_path = "data/train.csv"
valid_csv_path = "data/valid.csv"


def process_df(csv_path):
    df = pd.read_csv(csv_path)
    df["Path"] = "gs://cs221_chexpert/" + df["Path"].astype(str)
    dirname = os.path.dirname(csv_path)
    basename = os.path.basename(csv_path)
    output_path = os.path.join(dirname, "processed_{}".format(basename))
    df.to_csv(output_path, index=False)
    return df


process_df(train_csv_path)
process_df(valid_csv_path)
