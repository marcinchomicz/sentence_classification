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
TENSORBOARD_DIR = '/mnt/workdata/_WORK_/mail_zonning/mail_zoning/optim/sequenced_bilstm/tblogs/'
ENAME = 'SEQUENCED_ConvBiGRU_iter_3'

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


def split_dataset(data: pd.DataFrame, random_state: int):
    train_emails, test_emails = train_test_split(data['doc'].unique(), test_size=0.2, shuffle=True,
                                                 random_state=random_state)

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


mlflow.set_tracking_uri(MLFLOW_DIR)
try:
    eid = mlflow.get_experiment_by_name(ENAME).experiment_id
except:
    eid = mlflow.create_experiment(ENAME)

df = prepare_dataset(DATAPATH)
texts = df['sentence'].values

def define_model_conv_bilstm(model_params: dict):
    input_block = tf.keras.layers.Input(
        shape=(model_params['window_size'],
               model_params['output_sequence_length']),
        dtype=tf.int32,
        name='input_block')

    x = tf.keras.layers.Embedding(input_dim=len(vectorizer.get_vocabulary()),
                                  output_dim=model_params['embedding_dimension'],
                                  input_length=model_params['output_sequence_length'],
                                  mask_zero=False,
                                  name='embed')(input_block)

    x = tf.keras.layers.BatchNormalization(name='embed_batch_norm')(x)

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv1D(
            filters=model_params['conv1d_0_units'],
            kernel_size=model_params['conv1d_0_kernelsize'],
            padding=model_params['conv1d_0_padding'],
            activation=model_params['conv1d_0_activation'],
            name='conv1d_0'))(x)

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv1D(
            filters=model_params['conv1d_1_units'],
            kernel_size=model_params['conv1d_1_kernelsize'],
            padding=model_params['conv1d_1_padding'],
            activation=model_params['conv1d_1_activation'],
            name='conv1d_1'))(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(model_params['gru_0_units'], return_sequences=True)
    )(x)

    x = tf.keras.layers.Dropout(rate=model_params['drop_0_rate'])(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(model_params['gru_0_units'], return_sequences=True)
    )(x)

    x = tf.keras.layers.Dropout(rate=model_params['drop_0_rate'])(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(model_params['gru_0_units'], return_sequences=True)
    )(x)

    x = tf.keras.layers.Dropout(rate=model_params['drop_0_rate'])(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(model_params['gru_1_units'], return_sequences=False)
    )(x)

    x = tf.keras.layers.Dropout(rate=model_params['drop_1_rate'])(x)

    x = tf.keras.layers.Dense(
        units=3,
        activation='softmax')(x)

    model = tf.keras.models.Model(inputs=input_block, outputs=x)
    return model


def report_epoch_progress(epoch, logs):
    mlflow.log_metrics(logs)


def report_parameters(model_params: dict, model, model_definition, run):
    mlflow.log_params(params=model_params)
    mlflow.log_dict(model_params, 'model_params.txt')
    mlflow.log_text(json.dumps(json.loads(model.to_json()), indent=3), 'model_schema.txt')
    mlflow.log_text(inspect.getsource(model_definition), 'model_building_method.txt')
    if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
        mlflow.log_dict({'cuda_device': os.environ['CUDA_VISIBLE_DEVICES']}, 'enironment_settings.txt')
    diagram_filename = f"{FILESTORE}/diagram_{run.data.tags['mlflow.runName']}.png"
    img = tf.keras.utils.plot_model(model, to_file=diagram_filename, show_shapes=True, show_dtype=True,
                                    show_layer_names=True, show_layer_activations=True)
    mlflow.log_artifact(local_path=diagram_filename)
    os.remove(diagram_filename)


def report_metrics(y_true, y_pred, eval_loss):
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


'''@nni.variable(nni.uniform(1e-4, 6e-4), name=initial_lr)'''
initial_lr = 0.001
'''@nni.variable(nni.quniform(32, 68, 4), name=batch_size)'''
batch_size = 64
'''@nni.variable(nni.uniform(0.05, 0.95), name=lr_reduction_factor)'''
lr_reduction_factor = 0.7
'''@nni.variable(nni.quniform(2, 6, 1), name=lr_reduction_step)'''
lr_reduction_step = 2

model_params={
    "vocab_size": 8000,
    "output_sequence_length": 45,
    "embedding_dimension": 190,
    "window_size": 3,
    "conv1d_0_units": 56,
    "conv1d_0_kernelsize": 3,
    "conv1d_0_padding": "valid",
    "conv1d_0_activation": "relu",
    "conv1d_1_units": 96,
    "conv1d_1_kernelsize": 3,
    "conv1d_1_padding": "valid",
    "conv1d_1_activation": "relu",
    "gru_0_units": 48,
    "gru_1_units": 40,
    "dense_0_units": 104,
    "dense_0_activation": "relu",
    "drop_0_rate": 0.2600093538288592,
    "drop_1_rate": 0.2600093538288592,
    "initial_lr": 1e-3,
    "lr_reduction_factor": lr_reduction_factor,
    "lr_reduction_step": int(lr_reduction_step),

    "batch_size": int(batch_size),
    "max_epochs": 100,
    "early_stop_patience": 10

}

# add BOM and EOM signals to texts to include them in vocabulary
texts = np.append(texts, [BOM_SIGNAL, EOM_SIGNAL])

vectorizer = tf.keras.layers.TextVectorization(max_tokens=model_params['vocab_size'],
                                               output_sequence_length=model_params['output_sequence_length'],
                                               pad_to_max_tokens=True,
                                               output_mode='int',
                                               name='Vectorizer')
vectorizer.adapt(texts)

def lr_schedule(epoch, lr):
    if epoch>0 and epoch % model_params['lr_reduction_step'] ==0:
        return lr*model_params['lr_reduction_factor']
    else:
        return lr

model_definition = define_model_conv_bilstm
RANDOM_STATES = [123, 12, 42, 78, 95]
with mlflow.start_run(experiment_id=eid, nested=False, tags={'master': True}) as master_run:
    master_results = {}
    for k in [1,2,3]:
        for i in range(0, 2):
            tf.keras.backend.clear_session()
            rs = RANDOM_STATES[i]

            train_data, val_data = split_dataset(df, rs)
            train_texts, train_labels = prepare_records(train_data, vectorizer)
            val_texts, val_labels = prepare_records(val_data, vectorizer)

            model = model_definition(model_params)
            run_name=f"{master_run.data.tags['mlflow.runName']}-{k}-{rs}"
            with mlflow.start_run(experiment_id=eid,
                                  run_name=run_name, nested=True,
                                  tags={'master': False}) as run:
                model.compile(optimizer=tf.keras.optimizers.Adam(model_params['initial_lr']),
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=['accuracy'])
                report_parameters(model_params, model, model_definition, run)

                model.fit(x=train_texts,
                          y=train_labels,
                          batch_size=model_params['batch_size'],
                          epochs=model_params['max_epochs'],
                          validation_data=(val_texts, val_labels),
                          use_multiprocessing=True,
                          callbacks=[
                              tf.keras.callbacks.EarlyStopping(min_delta=1e-4,
                                                               patience=model_params['early_stop_patience'],
                                                               restore_best_weights=True),
                              tf.keras.callbacks.LearningRateScheduler(schedule=lr_schedule, verbose=1),
                              tf.keras.callbacks.LambdaCallback(on_epoch_end=report_epoch_progress),
                              tf.keras.callbacks.TensorBoard(log_dir=f'{TENSORBOARD_DIR}{run_name}/',write_graph=False)
                          ])

                eval_loss, eval_acc = model.evaluate(val_texts, val_labels)
                y_pred_ = model.predict(val_texts)
                y_true = val_labels
                y_pred = np.argmax(y_pred_, axis=1)
                master_results[i] = report_metrics(y_true, y_pred, eval_loss)
                final_metrics_ = {
                    'Loss_eval': eval_loss,
                    'default': accuracy_score(y_true, y_pred),
                }
                """@nni.report_intermediate_result(final_metrics_)"""
    report_parameters(model_params, model, model_definition,
                      master_run)  # model from the last iteration, params are shared among all of them
    master_results_ = report_master_results(master_results)
    master_results_ = {
        'default': master_results_['Accuracy'],
        'loss': master_results_['Loss_eval']
    }
    """@nni.report_final_result(master_results_)"""
