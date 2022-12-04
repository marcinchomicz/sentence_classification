import os
import sys
sys.path.append("/mnt/workdata/_WORK_/mail_zonning/mail_zoning/classes/")
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"

import mlflow
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import inspect
import sklearn
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sentence_classifier import MovingWindowSentenceClassifier

for m in [np, pd, tf, sklearn, mlflow]:
    print(f"{m.__name__:15s}\t{m.__version__}")

OUTPUTPATH = "/mnt/workdata/_WORK_/mail_zonning/mail_zoning/sandbox/"
DATAPATH = "/mnt/workdata/_WORK_/mail_zonning/mail_zoning/dataset/enron_files_annotated/"
FILESTORE = "/mnt/workdata/_WORK_/mail_zonning/mail_zoning/tmp/"
MLFLOW_DIR = "file:///home/chomima5/mlruns/"
ENAME = 'moving_window_ITER0'

BOM_SIGNAL = 'the start of the email signal, no lines before'
EOM_SIGNAL = 'the end of the email signal, no lines after'
RANDOM_STATES = [123]


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


def set_params():
    '''@nni.variable(nni.quniform(50, 300, 25), name=embedding_dimension)'''
    embedding_dimension = 200
    '''@nni.variable(nni.uniform(0., 0.5), name=drop_0_rate)'''
    drop_0_rate = 0.1

    '''@nni.variable(nni.uniform(0.05, 0.7), name=lr_reduction_factor)'''
    lr_reduction_factor = 0.1
    '''@nni.variable(nni.uniform(5e-5, 1e-3), name=initial_lr)'''
    initial_lr = 0.0005
    '''@nni.variable(nni.choice(16, 32, 64, 128, 256), name=batch_size)'''
    batch_size = 64

    model_params = {
        "output_class_count": 3,
        "vocab_size": 8000,
        "output_sequence_length": 45,
        "embedding_dimension": int(embedding_dimension),
        "window_size": 5,
        "conv1d_0_units": 64,
        "conv1d_0_kernelsize": 3,
        "conv1d_0_padding": "valid",
        "conv1d_0_activation": "relu",
        "conv1d_1_units": 64,
        "conv1d_1_kernelsize": 3,
        "conv1d_1_padding": "valid",
        "conv1d_1_activation": "relu",
        "gru_0_units": 128,
        "gru_1_units": 64,
        "drop_0_rate": drop_0_rate,

        "initial_lr": initial_lr,
        "lr_reduction_factor": lr_reduction_factor,
        "lr_reduction_patience": 5,
        "batch_size": batch_size,
        "max_epochs": 100,
        "early_stop_patience": 10
    }
    return model_params


def report_epoch_progress(epoch, logs):
    mlflow.log_metrics(logs)


def report_parameters(model_params: dict, model):
    mlflow.log_params(params=model_params)
    mlflow.log_dict(model_params, 'model_params.txt')
    mlflow.log_text(json.dumps(json.loads(model.to_json()), indent=3), 'model_schema.txt')
    if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
        mlflow.log_dict({'cuda_device': os.environ['CUDA_VISIBLE_DEVICES']}, 'enironment_settings.txt')
    diagram_filename = f"{FILESTORE}/diagram_{master_run.data.tags['mlflow.runName']}.png"
    img = tf.keras.utils.plot_model(model, to_file=diagram_filename, show_shapes=True, show_dtype=True,
                                    show_layer_names=True, show_layer_activations=True)
    mlflow.log_artifact(local_path=diagram_filename)
    os.remove(diagram_filename)
    summary_filename = f"{FILESTORE}/summary_{master_run.data.tags['mlflow.runName']}.txt"
    with open(summary_filename, 'w') as f:
        model.summary(expand_nested=True, print_fn=lambda x: f.write(x+"\n"))
    mlflow.log_artifact(local_path=summary_filename)
    os.remove(summary_filename)


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


mlflow.set_tracking_uri(MLFLOW_DIR)
try:
    eid = mlflow.get_experiment_by_name(ENAME).experiment_id
except:
    eid = mlflow.create_experiment(ENAME)
    print(f'Created mlflow experiment: {eid}')

RANDOM_STATE = 123
df = prepare_dataset(DATAPATH)
train_subsets, val_subsets = split_dataset(df, RANDOM_STATE)
texts = df['label'].values

label_count = df['label'].nunique()
model_params = set_params()

with mlflow.start_run(experiment_id=eid, nested=False, tags={'master': True}) as master_run:
    master_results = {}

    for idx in range(len(train_subsets)):
        tf.keras.backend.clear_session()
        with mlflow.start_run(experiment_id=eid,
                              run_name=f"{master_run.data.tags['mlflow.runName']}-{idx}", nested=True,
                              tags={'master': False}) as run:
            model = MovingWindowSentenceClassifier(
                model_params=model_params, corpus=df['sentence'].values, bod_line=BOM_SIGNAL, eod_line=EOM_SIGNAL)
            model.prepare_train_records(data=train_subsets[idx])
            model.prepare_validation_records(data=val_subsets[idx])
            model.compile(optimizer=tf.keras.optimizers.Adam(model_params['initial_lr']),
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=['accuracy'])
            report_parameters(model_params, model)

            model.fit(x=model.train_texts, y=model.train_labels,
                      batch_size=model_params['batch_size'],
                      epochs=model_params['max_epochs'],
                      validation_data=(model.validation_texts, model.validation_labels),
                      callbacks=[
                          tf.keras.callbacks.ReduceLROnPlateau(factor=model_params['lr_reduction_factor'],
                                                               patience=5,
                                                               verbose=1),
                          tf.keras.callbacks.EarlyStopping(min_delta=1e-4,
                                                           patience=10,
                                                           restore_best_weights=True),
                          tf.keras.callbacks.LambdaCallback(on_epoch_end=report_epoch_progress),
                          tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, baseline=0.9,
                                                           restore_best_weights=True)
                      ])
            eval_loss, eval_acc = model.evaluate(x=model.validation_texts, y=model.validation_labels)
            y_pred_ = model.predict(model.validation_texts)
            y_true = model.validation_labels
            y_pred = np.argmax(y_pred_, axis=1)
            master_results[idx] = report_metrics(y_true, y_pred)
            final_metrics_ = {
                'Loss_eval': eval_loss,
                'default': accuracy_score(y_true, y_pred),
            }
            """@nni.report_intermediate_result(master_results[idx])"""
    report_parameters(model_params, model)  # model from the last iteration, params are shared among all of them
    master_results_ = report_master_results(master_results)
    master_results_ = {
        'default': master_results_['Accuracy'],
        'loss': master_results_['Loss_eval']
    }
    """@nni.report_final_result(master_results_)"""
