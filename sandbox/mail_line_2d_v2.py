import os

os.environ['CUDA_VISIBLE_DEVICES'] = ""
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import mlflow
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import itertools
from tqdm import tqdm
import inspect

for m in [pd, tf, tfa, mlflow]:
    print(f"{m.__name__:15s}\t{m.__version__}")

OUTPUTPATH = "/mnt/workdata/_WORK_/mail_zonning/mail_zoning/sandbox/"
DATAPATH = "/mnt/workdata/_WORK_/mail_zonning/mail_zoning/dataset/enron_files_annotated/"
FILESTORE = "/mnt/workdata/_WORK_/mail_zonning/mail_zoning/tmp/"
MLFLOW_DIR = "file:///home/chomima5/mlruns/"
ENAME = 'Mail_Line_2D'

BOM_SIGNAL = 'the start of the email signal, no lines before'
EOM_SIGNAL = 'the end of the email signal, no lines after'

model_params = {
    'vocab_size': 8000,
    'output_sequence_length': 10,
    'window_size': 3,
    'initial_lr': 1e-4,
}

def prepare_dataset(datapath: str):
    def extract_text(text, flags: dict):
        text = text.split('\n')
        idx = 0
        while text[idx][:2] not in flags:
            idx += 1
        labels = [flags[t[:2]] for t in text[idx:] if len(t) > 1]
        text = [t[2:] for t in text[idx:]]
        return text, labels

    # load and extract flag data
    FLAGS = {'B>': 0, 'H>': 1, 'S>': 2}
    files = {}
    for filename in os.listdir(datapath):
        with open(os.path.join(datapath, filename), 'rt') as f:
            files[filename] = f.read()
    _ = []
    for filename in files.keys():
        text_ = files[filename]
        textlines, labels = extract_text(text_, FLAGS)
        for idx, line_label in enumerate(zip(textlines, labels)):
            _.append({'doc': filename, 'idx': idx, 'sentence': line_label[0], 'label': line_label[1]})
    df = pd.DataFrame.from_dict(_)

    return df


def split_dataset(data: pd.DataFrame):
    train_emails, test_emails = train_test_split(data['doc'].unique(), test_size=0.2, shuffle=True)

    train_data = data.loc[data['doc'].isin(train_emails)]
    train_data = train_data[['doc', 'idx', 'sentence', 'label']].values

    test_data = data.loc[data['doc'].isin(test_emails)]
    test_data = test_data[['doc', 'idx', 'sentence', 'label']].values

    return train_data, test_data


def prepare_records(data: np.ndarray, vectorizer):
    DOC = 0; IDX = 1; SENT = 2; LBL = 3
    texts = []
    labels_=[]
    for doc in np.unique(data[:, DOC]):
        sents = np.concatenate([np.array([BOM_SIGNAL]),
                                data[data[:, 0] == doc, SENT],
                                np.array([EOM_SIGNAL])])
        sents = vectorizer(sents).numpy()
        labels=data[data[:, 0] == doc, LBL]
        sents = sliding_window_view(
            x=sents,
            window_shape=(model_params['window_size'],
                          model_params['output_sequence_length']))
        for sent in sents:
            texts.append(sent)
        labels_.extend(labels)
    return np.squeeze(texts), np.stack(labels_)


df = prepare_dataset(DATAPATH)
train_data, val_data = split_dataset(df)
texts = df['sentence'].values

#add BOM and EOM signals to texts to include them in vocabulary
texts = np.append(texts, [BOM_SIGNAL, EOM_SIGNAL])

vectorizer = tf.keras.layers.TextVectorization(max_tokens=model_params['vocab_size'],
                                               output_sequence_length=model_params['output_sequence_length'],
                                               pad_to_max_tokens=True,
                                               output_mode='int',
                                               name='Vectorizer')
vectorizer.adapt(texts)

train_texts, train_labels = prepare_records(train_data, vectorizer)
val_texts, val_labels = prepare_records(val_data, vectorizer)

# Model
tf.keras.backend.clear_session()

input_block = tf.keras.layers.Input(
    shape=(model_params['window_size'], model_params['output_sequence_length'],),
    dtype=tf.int32, name='input_block')
x = tf.keras.layers.TimeDistributed(
    layer=tf.keras.layers.Embedding(input_dim=len(vectorizer.get_vocabulary()),
                                    output_dim=model_params['output_sequence_length'],
                                    mask_zero=True, name='embed'))(input_block)

x = tf.keras.layers.TimeDistributed(tf.keras.layers.LSTM(units=64))(x)
x = tf.keras.layers.Dense(units=64, activation='relu')(x)
x = tf.keras.layers.Dense(units=3, activation='softmax')(x)
#
# # na wyjściu bierzemy tylko element wynik dla elementu środkowegoboczkar
model = tf.keras.models.Model(inputs=input_block, outputs=x[:,1])
#
model.compile(optimizer=tf.keras.optimizers.Adam(model_params['initial_lr']),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(x=train_texts,
          y=train_labels,
          batch_size=16,
          epochs=100,
          validation_data=(val_texts, val_labels),
          callbacks=[
              tf.keras.callbacks.ReduceLROnPlateau(factor=0.1,
                                                   patience=5,
                                                   verbose=1),
              tf.keras.callbacks.EarlyStopping(min_delta=1e-4,
                                               patience=10,
                                               restore_best_weights=True),

          ])
