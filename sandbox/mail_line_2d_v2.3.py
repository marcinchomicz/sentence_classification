import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

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

# policy = tf.keras.mixed_precision.Policy('mixed_float16')
# tf.keras.mixed_precision.set_global_policy(policy)
# print('Compute dtype: %s' % policy.compute_dtype)
# print('Variable dtype: %s' % policy.variable_dtype)

tf.config.experimental.enable_tensor_float_32_execution(enabled=True)

OUTPUTPATH = "/mnt/workdata/_WORK_/mail_zonning/mail_zoning/sandbox/"
DATAPATH = "/mnt/workdata/_WORK_/mail_zonning/mail_zoning/dataset/enron_files_annotated/"
FILESTORE = "/mnt/workdata/_WORK_/mail_zonning/mail_zoning/tmp/"
MLFLOW_DIR = "file:///home/chomima5/mlruns/"
ENAME = 'Mail_Line_2D'

BOM_SIGNAL = 'the start of the email signal, no lines before'
EOM_SIGNAL = 'the end of the email signal, no lines after'

model_params = {
    'vocab_size': 8000,
    'output_sequence_length': 45,
    'embedding_dimension':200,
    'window_size': 3,

    'bilstm_0_units': 128,
    'bilstm_1_units': 128,
    'conv_1_units': 64,
    'conv_2_units': 64,
    'dense_0_units': 32,

    'initial_lr': 1e-4,
    'batch_size': 64,
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
    DOC = 0;
    IDX = 1;
    SENT = 2;
    LBL = 3
    texts = []
    labels_ = []
    for doc in np.unique(data[:, DOC]):
        sents = np.concatenate([np.array([BOM_SIGNAL]),
                                data[data[:, 0] == doc, SENT],
                                np.array([EOM_SIGNAL])])
        sents = vectorizer(sents).numpy()
        labels = data[data[:, 0] == doc, LBL]
        sents = sliding_window_view(
            x=sents,
            window_shape=(model_params['window_size'],
                          model_params['output_sequence_length']))
        for sent in sents:
            texts.append(sent)
        labels_.extend(labels)
    return np.squeeze(texts), np.stack(labels_)

def prepare_records_2(data: np.ndarray, vectorizer):
    DOC = 0;
    IDX = 1;
    SENT = 2;
    LBL = 3
    texts = []
    labels_ = []
    for doc in np.unique(data[:, DOC]):
        sents = np.concatenate([np.array([BOM_SIGNAL]),
                                data[data[:, 0] == doc, SENT],
                                np.array([EOM_SIGNAL])])
        sents = vectorizer(sents).numpy()
        # labels = data[data[:, 0] == doc, LBL]
        labels = np.concatenate([np.array([0]),
                                data[data[:, 0] == doc,LBL],
                                np.array([0])])
        for sent in sents:
            texts.append(sent)
        labels_.extend(labels)
    return np.array(texts), np.array(labels_)

df = prepare_dataset(DATAPATH)
train_data, val_data = split_dataset(df)
texts = df['sentence'].values

# add BOM and EOM signals to texts to include them in vocabulary
texts = np.append(texts, [BOM_SIGNAL, EOM_SIGNAL])

vectorizer = tf.keras.layers.TextVectorization(max_tokens=model_params['vocab_size'],
                                               output_sequence_length=model_params['output_sequence_length'],
                                               pad_to_max_tokens=True,
                                               output_mode='int',
                                               name='Vectorizer')
vectorizer.adapt(texts)

train_texts, train_labels = prepare_records(train_data, vectorizer)
val_texts, val_labels = prepare_records(val_data, vectorizer)
# train_texts, train_labels = prepare_records_2(train_data, vectorizer)
# val_texts, val_labels = prepare_records_2(val_data, vectorizer)


# Model
def define_model_bilstm(model_params: dict):
    input_block = tf.keras.layers.Input(
        shape=(model_params['window_size'],
               model_params['output_sequence_length'],),
        dtype=tf.int32, name='input_block')

    x = tf.keras.layers.TimeDistributed(
        layer=tf.keras.layers.Embedding(input_dim=len(vectorizer.get_vocabulary()),
                                        output_dim=model_params['output_sequence_length'],
                                        mask_zero=True,
                                        name='embed'))(input_block)

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=model_params['bilstm_0_units'], return_sequences=True, name='BiLSTM_0')))(x)
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=model_params['bilstm_1_units'], return_sequences=False, name='BiLSTM_1')))(x)

    x = tf.keras.layers.Dense(units=model_params['dense_0_units'], activation='relu', name='Dense_0')(x)
    x = tf.keras.layers.Dense(units=3, activation='softmax', name='dense_final')(x)
    #
    # # na wyjściu bierzemy tylko element wynik dla elementu środkowegoboczkar
    model = tf.keras.models.Model(inputs=input_block, outputs=x[:, 1])
    #
    model.compile(optimizer=tf.keras.optimizers.Adam(model_params['initial_lr']),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model

def define_model_bilstm_cnn(model_params: dict):
    input_block = tf.keras.layers.Input(
        shape=(model_params['window_size'],
               model_params['output_sequence_length'],),
        dtype=tf.int32, name='input_block')

    x = tf.keras.layers.TimeDistributed(
        layer=tf.keras.layers.Embedding(input_dim=len(vectorizer.get_vocabulary()),
                                        output_dim=model_params['output_sequence_length'],
                                        mask_zero=True,
                                        name='embed'))(input_block)

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=model_params['bilstm_0_units'], return_sequences=True, name='BiLSTM_0')))(x)
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=model_params['bilstm_1_units'], return_sequences=False, name='BiLSTM_1')))(x)
    x = tf.keras.layers.Dense(units=model_params['dense_0_units'], activation='relu', name='Dense_0')(x)
    x = tf.keras.layers.Dense(units=3, activation='softmax', name='dense_final')(x)
    #
    # # na wyjściu bierzemy tylko element wynik dla elementu środkowegoboczkar
    model = tf.keras.models.Model(inputs=input_block, outputs=x[:, 1])
    #
    model.compile(optimizer=tf.keras.optimizers.Adam(model_params['initial_lr']),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model

def define_model_cnn(model_params: dict):
    input_block = tf.keras.layers.Input(
        shape=(model_params['window_size'],
               model_params['output_sequence_length'],),
        dtype=tf.int32,
        name='Input')
    print(f"Input block: {input_block.shape}")
    x = tf.keras.layers.TimeDistributed(
        layer=tf.keras.layers.Embedding(input_dim=len(vectorizer.get_vocabulary()),
                                        output_dim=model_params['embedding_dimension'],
                                        mask_zero=True,
                                        name='Embed'))(input_block)
    print(f"After embedding: {x.shape}")

    # x=tf.keras.layers.Conv2D(filters=64, kernel_size=3)(x)
    # print(f"After Conv2D: {x.shape}")

    x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units=model_params['bilstm_0_units'],  return_sequences=True, name='BiLSTM_0'),
            name='TS_BiLSTM_0', input_shape=()))(x)
    print(f"After BiLSTM_0: {x.shape}")

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=model_params['bilstm_1_units'], return_sequences=False, name='BiLSTM_1'), name='TS_BiLSTM_1'))(x)
    print(f"After BiLSTM_1: {x.shape}")

    x = tf.keras.layers.Dense(units=model_params['dense_0_units'], activation='relu', name='Dense_0')(x)
    print(f"After Dense_0: {x.shape}")
    x = tf.keras.layers.Dense(units=3, activation='softmax', name='Dense_FIN')(x)
    print(f"After Dense_final: {x.shape}")
    #
    # # na wyjściu bierzemy tylko element wynik dla elementu środkowegoboczkar

    model = tf.keras.models.Model(inputs=input_block, outputs=x[:,1])
    print(f"On model: {x[:,1].shape}")
    #
    model.compile(optimizer=tf.keras.optimizers.Adam(model_params['initial_lr']),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model

def define_model_bilstm_2(model_params: dict):
    input_block = tf.keras.layers.Input(
        shape=(model_params['window_size'],
               model_params['output_sequence_length']),
        dtype=tf.int32,
        # type_spec=tf.TensorSpec(shape=[model_params['window_size'], model_params['output_sequence_length']], dtype=tf.int32),

        name='input_block')
    print(f"After input block: {input_block.shape}")
    x = tf.keras.layers.Embedding(input_dim=len(vectorizer.get_vocabulary()),
                                        output_dim=model_params['embedding_dimension'],
                                        mask_zero=True,
                                        name='embed')(input_block)
    print(f"After embedding: {x.shape}")

    x= tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=model_params['bilstm_0_units'], return_sequences=True, name='BiLSTM_0')))(x)
    print(f"After BiLSTM_0: {x.shape}")

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=model_params['bilstm_1_units'], return_sequences=False, name='BiLSTM_1')))(x)
    print(f"After BiLSTM_1: {x.shape}")


    # x = tf.keras.layers.Conv1D(filters=model_params['conv_1_units'], kernel_size=3, activation='relu')(x)
    # print(f"After Conv1D_1: {x.shape}")
    #
    # x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    # print(f"After MaxPool_1: {x.shape}")

    x = tf.keras.layers.Dense(units=model_params['dense_0_units'], activation='relu', name='Dense_0')(x)
    print(f"After Dense_0: {x.shape}")

    x = tf.keras.layers.Dense(units=3, activation='softmax', name='dense_final')(x)
    print(f"After Dense_FIN: {x.shape}")

    # # na wyjściu bierzemy tylko element wynik dla elementu środkowegoboczkar
    model = tf.keras.models.Model(inputs=input_block, outputs=x[:,1])
    model.compile(optimizer=tf.keras.optimizers.Adam(model_params['initial_lr']),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model

def define_model_bilstm_3(model_params: dict):
    input_block = tf.keras.layers.Input(
        shape=(model_params['window_size'],
               model_params['output_sequence_length']),
        dtype=tf.int32,
        # type_spec=tf.TensorSpec(shape=[model_params['window_size'], model_params['output_sequence_length']], dtype=tf.int32),

        name='input_block')
    print(f"After input block: {input_block.shape}")
    x = tf.keras.layers.Embedding(input_dim=len(vectorizer.get_vocabulary()),
                                        output_dim=model_params['embedding_dimension'],
                                        mask_zero=True,
                                        name='embed')(input_block)
    print(f"After embedding: {x.shape}")

    x= tf.keras.layers.BatchNormalization()(x)
    print(f"After batch normalization: {x.shape}")

    x =tf.keras.layers.TimeDistributed(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=model_params['bilstm_0_units'], return_sequences=True, name='BiLSTM_0')))(x)
    print(f"After BiLSTM_0: {x.shape}")

    # x = tf.keras.layers.TimeDistributed(
    #     tf.keras.layers.Bidirectional(
    #         tf.keras.layers.LSTM(units=model_params['bilstm_1_units'], return_sequences=False, name='BiLSTM_1')))(x)
    # print(f"After BiLSTM_1: {x.shape}")


    # x = tf.keras.layers.Conv1D(filters=model_params['conv_1_units'], kernel_size=3, activation='relu')(x)
    # print(f"After Conv1D_1: {x.shape}")
    #
    # x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    # print(f"After MaxPool_1: {x.shape}")

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(units=model_params['dense_0_units'], activation='relu', name='Dense_0'))(x)
    print(f"After Dense_0: {x.shape}")

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=3, activation='softmax', name='dense_final'))(x)
    print(f"After Dense_FIN: {x.shape}")

    # # na wyjściu bierzemy tylko element wynik dla elementu środkowegoboczkar
    model = tf.keras.models.Model(inputs=input_block, outputs=x[:,1])
    model.compile(optimizer=tf.keras.optimizers.Adam(model_params['initial_lr']),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model

def define_model_bilstm_4(model_params: dict):
    input_block = tf.keras.layers.Input(
        shape=(model_params['window_size'],
               model_params['output_sequence_length']),
        dtype=tf.int32,
        name='input_block')
    x = tf.keras.layers.Embedding(input_dim=len(vectorizer.get_vocabulary()),
                                  output_dim=model_params['embedding_dimension'],
                                  input_length=model_params['output_sequence_length'],
                                  mask_zero=True,
                                  name='embed')(input_block)


    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(units=128, return_sequences=True, name='BiLSTM_0')))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(units=128, return_sequences=False, name='BiLSTM_1')))(x)

    x = tf.keras.layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(units=64, activation='relu', name='Dense_0'))(x)

    x = tf.keras.layers.Dense(units=3, activation='softmax', name='dense_final')(x)

    model = tf.keras.models.Model(inputs=input_block, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(model_params['initial_lr']),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model

def lr_scheduler(epoch, lr):
    return lr * 1 / (epoch + 1)
    # if epoch % 2 ==0:
    #     return lr
    # else:
    #     return lr*0.5


tf.keras.backend.clear_session()
model = define_model_bilstm_4(model_params)
tf.keras.utils.plot_model(model, to_file="/mnt/workdata/_WORK_/mail_zonning/mail_zoning/sandbox/_model.png",
                          show_shapes=True,show_dtype=True,show_layer_names=True, show_layer_activations=True,expand_nested=True)
model.fit(x=train_texts,
          y=train_labels,
          batch_size=model_params['batch_size'],
          epochs=100,
          validation_data=(val_texts, val_labels),
          use_multiprocessing=True,
          callbacks=[
              tf.keras.callbacks.ReduceLROnPlateau(factor=0.1,
                                                   patience=5,
                                                   verbose=1),
              tf.keras.callbacks.EarlyStopping(min_delta=1e-4,
                                               patience=10,
                                               restore_best_weights=True),
              # tf.keras.callbacks.LearningRateScheduler(schedule=lr_scheduler,
              #                                          verbose=True)

          ])
