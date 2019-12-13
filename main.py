from datetime import datetime
import tensorflow as tf
from data import build_dataset
from model import get_model, get_global_model
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str, default = "ResNet50", help = 'model name')
    args = parser.parse_args()
    return args

def train(model_name, model):
    logdir = "logs/" + model_name + "/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = "model/" + model_name + "/checkpoint.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    train_dataset = build_dataset(TRAIN_CSV_PATH).repeat().prefetch(256).shuffle(256).batch(32)
    valid_dataset = build_dataset(VALID_CSV_PATH).repeat().prefetch(100).shuffle(100).batch(16)
    model.fit(train_dataset, epochs=300, steps_per_epoch=10, validation_steps=3, validation_data=valid_dataset, callbacks= [tensorboard_callback, cp_callback])


TRAIN_CSV_PATH = "data/data/processed_train.csv"
VALID_CSV_PATH = "data/data/processed_valid.csv"
args = get_args()

# python3 main.py --model ResNet50
if args.model == "ResNet50":
    print("You chose to use ResNet50.")
    TRAIN_CSV_PATH = "data/data/processed_train.csv"
    VALID_CSV_PATH = "data/data/processed_valid.csv"
    train("ResNet50", get_model())
    
# python3 main.py --model AG-CNN
elif args.model == "AG-CNN":
    print("You chose to use AG-CNN.")
    TRAIN_CSV_PATH = "data/data/processed_attention_train.csv"
    VALID_CSV_PATH = "data/data/processed_attention_valid.csv"
    train("AG-CNN", get_global_model())

# python3 main.py --model AG-CNN2
elif args.model == "AG-CNN2":
    print("You chose to use AG-CNN.")
    TRAIN_CSV_PATH = "data/data/processed_attention_train.csv"
    VALID_CSV_PATH = "data/data/processed_attention_valid.csv"
    train("AG-CNN2", get_global_model())
    
# python3 main.py --model AG-CNN-FINAL
elif args.model == "AG-CNN-FINAL":
    print("You chose to use AG-CNN-FINAL.")
    TRAIN_CSV_PATH = "data/data/processed_attention_train.csv"
    VALID_CSV_PATH = "data/data/processed_attention_valid.csv"
    train("AG-CNN-FINAL", get_global_model())
