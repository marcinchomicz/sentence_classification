import os

os.environ['CUDA_VISIBLE_DEVICES'] = ""
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
import numpy as np
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
import nni

for m in [pd, tf, tfa, mlflow]:
    print(f"{m.__name__:15s}\t{m.__version__}")

OUTPUTPATH = "/mnt/workdata/_WORK_/mail_zonning/mail_zoning/sandbox/"
DATAPATH = "/mnt/workdata/_WORK_/mail_zonning/mail_zoning/dataset/enron_files_annotated/"
FILESTORE = "/mnt/workdata/_WORK_/mail_zonning/mail_zoning/tmp/"
MLFLOW_DIR = "file:///home/chomima5/mlruns/"
ENAME = 'PS_Context-BiLSTM-CNN'

BOM_SIGNAL = 'the start of the email signal, no lines before'
EOM_SIGNAL = 'the end of the email signal, no lines after'


# policy = tf.keras.mixed_precision.Policy('mixed_float16')
# tf.keras.mixed_precision.set_global_policy(policy)

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

    # add predecessors and followers for each line
    df['preceding'] = ''
    df['following'] = ''

    df = df[['doc', 'idx', 'preceding', 'sentence', 'following', 'label']]
    for doc in df['doc'].unique():
        _ = df.loc[df['doc'] == doc, 'sentence']
        df.loc[df['doc'] == doc, 'preceding'] = _.shift(periods=1, fill_value=BOM_SIGNAL)
        df.loc[df['doc'] == doc, 'following'] = _.shift(periods=-1, fill_value=EOM_SIGNAL)
    return df


def split_dataset(data: pd.DataFrame, random_state: int):
    train_emails, test_emails = train_test_split(data['doc'].unique(), test_size=0.2, shuffle=True,
                                                 random_state=random_state)

    train_data = data.loc[data['doc'].isin(train_emails)]
    train_labels = train_data['label'].values
    train_data = train_data[['preceding', 'sentence', 'following']].values

    test_data = data.loc[data['doc'].isin(test_emails)]
    test_labels = test_data['label'].values
    test_data = test_data[['preceding', 'sentence', 'following']].values

    return train_data, train_labels, test_data, test_labels


df = prepare_dataset(DATAPATH)


mlflow.set_tracking_uri(MLFLOW_DIR)
try:
    eid = mlflow.get_experiment_by_name(ENAME).experiment_id
except:
    eid = mlflow.create_experiment(ENAME)


def define_model_context_bilstm_cnn(adapted_text_vectorization_layer, model_params: dict, label_count: int):
    tf.keras.backend.clear_session()

    # Input layer is required, datatype and shape must be provided as well
    input_pred = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name='input_preceding')
    input_sent = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name='input_sentence')
    input_post = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name='input_following')

    x_pred = adapted_text_vectorization_layer(input_pred)
    x_sent = adapted_text_vectorization_layer(input_sent)
    x_post = adapted_text_vectorization_layer(input_post)

    embed_pred = tf.keras.layers.Embedding(input_dim=len(adapted_text_vectorization_layer.get_vocabulary()),
                                           output_dim=model_params['embedding_dimension'],
                                           mask_zero=True, name='pred_embed')(x_pred)
    embed_sent = tf.keras.layers.Embedding(input_dim=len(adapted_text_vectorization_layer.get_vocabulary()),
                                           output_dim=model_params['embedding_dimension'],
                                           mask_zero=True, name='sent_embed')(x_sent)
    embed_post = tf.keras.layers.Embedding(input_dim=len(adapted_text_vectorization_layer.get_vocabulary()),
                                           output_dim=model_params['embedding_dimension'],
                                           mask_zero=True, name='post_embed')(x_post)

    dense_pred = tf.keras.layers.Dense(units=model_params['dense_0_pred_size'], activation='relu', name='pred_dense_0')(
        embed_pred)
    dense_pred = tf.keras.layers.Flatten()(dense_pred)
    dense_pred = tf.keras.layers.Dropout(rate=model_params['drop_pred_rate'], name='pred_drop')(dense_pred)
    dense_pred = tf.keras.layers.Dense(units=model_params['dense_1_pred_size'], activation='relu', name='pred_dense_1')(
        dense_pred)

    dense_post = tf.keras.layers.Dense(units=model_params['dense_0_post_size'], activation='relu', name='post_dense_0')(
        embed_post)
    dense_post = tf.keras.layers.Flatten()(dense_post)
    dense_post = tf.keras.layers.Dropout(rate=model_params['drop_post_rate'], name='post_drop')(dense_post)
    dense_post = tf.keras.layers.Dense(units=model_params['dense_1_post_size'], activation='relu', name='post_dense_1')(
        dense_post)

    sent = tf.keras.layers.Bidirectional(
        layer=tf.keras.layers.LSTM(units=model_params['bilstm_sent_units'], return_sequences=True), name='sent_bilstm')(
        embed_sent)
    sent = tf.keras.layers.Convolution1D(filters=model_params['conv1D_size'],
                                         kernel_size=model_params['conv1D_kernel_size'],
                                         strides=1, padding='valid', activation='relu', name='conv1d_1')(sent)
    sent = tf.keras.layers.MaxPooling1D(pool_size=model_params['pool1d_pool_size'], name='maxpool_1')(sent)
    sent = tf.keras.layers.Flatten()(sent)
    sent = tf.keras.layers.Dense(units=model_params['dense_sent_size'],
                                 activation=model_params['dense_sent_activation'], name='sent_dense')(sent)
    merged = tf.keras.layers.Concatenate(axis=1, name='merger')([dense_pred, sent, dense_post])

    merged = tf.keras.layers.Dropout(rate=model_params['merger_dropout_rate'], name='merger_dropout')(merged)

    merged = tf.keras.layers.Dense(units=model_params['dense_merger_size'],
                                   activation=model_params['dense_merger_activation'], name='merged_dense')(merged)
    output_layer = tf.keras.layers.Dense(units=df['label'].nunique(), use_bias=True, activation='softmax',
                                         name='Output')(merged)

    model = tf.keras.models.Model(inputs=[input_pred, input_sent, input_post], outputs=output_layer)

    return model


'''@nni.variable(nni.quniform(50, 300, 25), name=embedding_dimension)'''
embedding_dimension = 200
'''@nni.variable(nni.quniform(16, 128, 16), name=dense_0_pred_size)'''
dense_0_pred_size = 64
'''@nni.variable(nni.uniform(0., 0.5), name=drop_pred_rate)'''
drop_pred_rate = 0.1
'''@nni.variable(nni.quniform(16, 128, 16), name=dense_1_pred_size)'''
dense_1_pred_size = 64
'''@nni.variable(nni.quniform(16, 128, 16), name=dense_0_post_size)'''
dense_0_post_size = 64
'''@nni.variable(nni.uniform(0., 0.5), name=drop_post_rate)'''
drop_post_rate = 0.1
'''@nni.variable(nni.quniform(16, 128, 16), name=dense_1_post_size)'''
dense_1_post_size = 64
'''@nni.variable(nni.choice(16, 32, 48, 64, 96, 128), name=bilstm_sent_units)'''
bilstm_sent_units = 128
'''@nni.variable(nni.choice(16, 32, 48, 64, 96, 128), name=conv1d_size)'''
conv1d_size = 32
'''@nni.variable(nni.quniform(16, 128, 16), name=dense_sent_size)'''
dense_sent_size = 128
'''@nni.variable(nni.uniform(0., 0.5), name=merger_dropout_rate)'''
merger_dropout_rate = 0.5
'''@nni.variable(nni.quniform(16, 128, 16), name=dense_merger_size)'''
dense_merger_size = 32
'''@nni.variable(nni.uniform(0.05, 0.7), name=lr_reduction_factor)'''
lr_reduction_factor = 0.1
'''@nni.variable(nni.uniform(5e-5, 1e-3), name=initial_lr)'''
initial_lr = 0.0005
'''@nni.variable(nni.choice(16,32,64), name=batch_size)'''
batch_size = 64

mp_context_bilstm_cnn = {
    'vocab_size': 8000,
    'output_sequence_length': 45,
    'embedding_dimension': int(embedding_dimension),
    'dense_0_pred_size': int(dense_0_pred_size),
    'drop_pred_rate': drop_pred_rate,
    'dense_1_pred_size': int(dense_1_pred_size),
    'dense_0_post_size': int(dense_0_post_size),
    'drop_post_rate': drop_post_rate,
    'dense_1_post_size': int(dense_1_post_size),
    'bilstm_sent_units': int(bilstm_sent_units),
    'conv1D_size': int(conv1d_size),
    'conv1D_kernel_size': 3,
    'pool1d_pool_size': 2,
    'dense_sent_size': int(dense_sent_size),
    'dense_sent_activation': 'relu',
    'merger_dropout_rate': merger_dropout_rate,
    'dense_merger_size': int(dense_merger_size),
    'dense_merger_activation': 'relu',
    'initial_lr': initial_lr,
    'max_epochs': 100,
    'lr_reduction_factor': lr_reduction_factor,
    'batch_size': int(batch_size)
}


def report_epoch_progress(epoch, logs):
    mlflow.log_metrics(logs)


def report_parameters(model_params: dict, model, model_definition):
    mlflow.log_params(params=model_params)
    mlflow.log_dict(model_params, 'model_params.txt')
    mlflow.log_text(json.dumps(json.loads(model.to_json()), indent=3), 'model_schema.txt')
    mlflow.log_text(inspect.getsource(model_definition), 'model_building_method.txt')
    if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
        mlflow.log_dict({'cuda_device': os.environ['CUDA_VISIBLE_DEVICES']}, 'enironment_settings.txt')
    diagram_filename = f"{FILESTORE}/diagram_{master_run.data.tags['mlflow.runName']}.png"
    img = tf.keras.utils.plot_model(model, to_file=diagram_filename, show_shapes=True, show_dtype=True,
                                    show_layer_names=True, show_layer_activations=True)
    mlflow.log_artifact(local_path=diagram_filename)
    os.remove(diagram_filename)


def report_metrics(y_true, y_pred):
    final_metrics = {
        'Loss_eval': eval_loss,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1-score': f1_score(y_true, y_pred, average='weighted')
    }
    mlflow.log_metrics(final_metrics)
    mlflow.log_dict(classification_report(y_true, y_pred, output_dict=True), 'classification_report.txt')
    return final_metrics


def report_master_results(master_results: dict):
    mlflow.log_dict(master_results, 'iteration_results.txt')
    master_results_ = pd.DataFrame(master_results).mean(axis=1).to_dict()
    mean_vals = pd.DataFrame(master_results).mean(axis=1)
    std_vals = pd.DataFrame(master_results).std(axis=1)
    mean_vals.index = [f"{c}_mean" for c in mean_vals.index]
    std_vals.index = [f"{c}_std" for c in std_vals.index]
    _ = {**mean_vals.to_dict(), **std_vals.to_dict()}
    mlflow.log_metrics(_)
    return master_results_

def prepare_vectorizer(data:pd.DataFrame, model_params:dict):
    texts = data['sentence'].values
    texts = np.append(texts, [BOM_SIGNAL, EOM_SIGNAL])

    vectorizer = tf.keras.layers.TextVectorization(max_tokens=model_params['vocab_size'],
                                                   output_sequence_length=model_params['output_sequence_length'],
                                                   pad_to_max_tokens=True,
                                                   output_mode='int',
                                                   name='Vectorizer')
    vectorizer.adapt(texts)
    return vectorizer


model_params = mp_context_bilstm_cnn
model_definition = define_model_context_bilstm_cnn
label_count = df['label'].nunique()


RANDOM_STATES=[123, 12, 42]

with mlflow.start_run(experiment_id=eid, nested=False, tags={'master': True}) as master_run:
    master_results = {}

    for i in range(0, 3):
        rs=RANDOM_STATES[i]
        train_data, train_labels, test_data, test_labels = split_dataset(df, rs)
        tf.keras.backend.clear_session()
        with mlflow.start_run(experiment_id=eid,
                              run_name=f"{master_run.data.tags['mlflow.runName']}-{i}-{rs}", nested=True,
                              tags={'master': False}) as run:

            vectorization_layer = prepare_vectorizer(df, model_params)
            model = model_definition(vectorization_layer, model_params, label_count)
            model.compile(optimizer=tf.keras.optimizers.Adam(model_params['initial_lr']),
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=['accuracy'])

            report_parameters(model_params, model, model_definition)
            model.fit(x=(train_data[:, 0], train_data[:, 1], train_data[:, 2]),
                      y=train_labels,
                      batch_size=model_params['batch_size'],
                      epochs=model_params['max_epochs'],
                      validation_data=((test_data[:, 0], test_data[:, 1], test_data[:, 2]), test_labels),
                      callbacks=[
                          tf.keras.callbacks.ReduceLROnPlateau(factor=model_params['lr_reduction_factor'],
                                                               patience=5,
                                                               verbose=1),
                          tf.keras.callbacks.EarlyStopping(min_delta=1e-4,
                                                           patience=10,
                                                           restore_best_weights=True),
                          tf.keras.callbacks.LambdaCallback(on_epoch_end=report_epoch_progress)
                      ])
            eval_loss, eval_acc = model.evaluate((test_data[:, 0], test_data[:, 1], test_data[:, 2]), test_labels)
            y_pred_ = model.predict((test_data[:, 0], test_data[:, 1], test_data[:, 2]))
            y_true = test_labels
            y_pred = np.argmax(y_pred_, axis=1)
            master_results[i] = report_metrics(y_true, y_pred)
            final_metrics_ = {
                'Loss_eval': eval_loss,
                'default': accuracy_score(y_true, y_pred),
            }
            """@nni.report_intermediate_result(master_results[i])"""
    report_parameters(model_params, model, model_definition) # model from the last iteration, params are shared among all of them
    master_results_ = report_master_results(master_results)
    master_results_ = {
        'default': master_results_['Accuracy'],
        'loss': master_results_['Loss_eval']
    }
    """@nni.report_final_result(master_results_)"""
