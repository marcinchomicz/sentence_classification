import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sentence_classifier import MovingWindowSentenceClassifier

OUTPUTPATH = "/mnt/workdata/_WORK_/mail_zonning/mail_zoning/sandbox/"
DATAPATH = "/mnt/workdata/_WORK_/mail_zonning/mail_zoning/dataset/enron_files_annotated/"
FILESTORE = "/mnt/workdata/_WORK_/mail_zonning/mail_zoning/tmp/"
MLFLOW_DIR = "file:///mnt/workdata/_WORK_/mail_zonning/mail_zoning/mlruns/"
TENSORBOARD_DIR = '/mnt/workdata/_WORK_/mail_zonning/mail_zoning/optim/sequenced_bilstm/tblogs/'
ENAME = 'SEQUENCED_CBiGRU_iter_0'

BOM_SIGNAL = 'the start of the email signal, no lines before'
EOM_SIGNAL = 'the end of the email signal, no lines after'


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
    flags = {'B>': 0, 'H>': 1, 'S>': 2}
    files = {}
    for filename in os.listdir(datapath):
        with open(os.path.join(datapath, filename), 'rt') as f:
            files[filename] = f.read()
    _ = []
    for filename in files.keys():
        text_ = files[filename]
        textlines, labels = extract_text(text_, flags)
        for idx, line_label in enumerate(zip(textlines, labels)):
            _.append({'doc': filename, 'idx': idx, 'sentence': line_label[0], 'label': line_label[1]})
    df = pd.DataFrame(_)
    return df


def split_dataset(data: pd.DataFrame, random_state: int):
    """"
    Dataset split is based on complete emails, not on email lines
    """
    splitter = KFold(n_splits=5, shuffle=True, random_state=random_state)
    docs = data['doc'].unique()
    splits = splitter.split(docs)
    train_data = []
    val_data = []
    for train_idx, val_idx in splits:
        train_data_ = data.loc[data['doc'].isin(docs[train_idx])]
        train_data.append(train_data_[['doc', 'idx', 'sentence', 'label']])
        val_data_ = data.loc[data['doc'].isin(docs[val_idx])]
        val_data.append(val_data_[['doc', 'idx', 'sentence', 'label']])
    return train_data, val_data


model_params = {
    "output_class_count": 3,
    "vocab_size": 8000,
    "output_sequence_length": 45,
    "embedding_dimension": 150,
    "window_size": 3,
    "conv1d_0_units": 80,
    "conv1d_0_kernelsize": 3,
    "conv1d_0_padding": "valid",
    "conv1d_0_activation": "relu",
    "conv1d_1_units": 80,
    "conv1d_1_kernelsize": 3,
    "conv1d_1_padding": "valid",
    "conv1d_1_activation": "relu",
    "gru_0_units": 128,
    "gru_1_units": 64,
    "drop_0_rate": 0.3448836829019953,
    "initial_lr": 0.0006500000000000001,
    "lr_reduction_factor": 0.44531593652922397,
    "lr_reduction_patience": 3,
    "batch_size": 56,
    "max_epochs": 100,
    "early_stop_patience": 10
}
print(model_params.keys())
# %%
RANDOM_STATE = 123
df = prepare_dataset(DATAPATH)
train_subsets, val_subsets = split_dataset(df, RANDOM_STATE)

texts = df['sentence'].values
# %%
results = {}
for idx in range(len(train_subsets)):
    tf.keras.backend.clear_session()
    clf = MovingWindowSentenceClassifier(
        model_params=model_params,
        bod_line='This is the first line of document. No lines come before.',
        eod_line='This is the last line of document. No lines come after.',
        corpus=texts)
    clf.prepare_train_records(data=train_subsets[idx])
    clf.prepare_validation_records(data=val_subsets[idx])
    clf.compile(optimizer=tf.keras.optimizers.Adam(model_params['initial_lr']),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])
    clf.fit(
        x=clf.train_texts, y=clf.train_labels,
        batch_size=model_params['batch_size'],
        epochs=model_params['max_epochs'],
        validation_data=(clf.validation_texts, clf.validation_labels),
        use_multiprocessing=True,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=model_params['lr_reduction_factor'], patience=model_params['lr_reduction_patience'],
                verbose=0),
            tf.keras.callbacks.EarlyStopping(
                min_delta=1e-4, patience=model_params['early_stop_patience'], restore_best_weights=True)
        ], verbose=1)
    results[idx] = clf.evaluate(clf.validation_texts, clf.validation_labels)