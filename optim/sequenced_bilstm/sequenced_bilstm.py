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
ENAME = 'SEQUENCED_3_BiLSTM'

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


# Model
def define_model_bilstm(model_params: dict):
    input_block = tf.keras.layers.Input(
        shape=(model_params['window_size'],
               model_params['output_sequence_length'],),
        dtype=tf.int32, name='input_block')

    x = tf.keras.layers.TimeDistributed(
        layer=tf.keras.layers.Embedding(input_dim=len(vectorizer.get_vocabulary()),
                                        output_dim=model_params['embedding_dimension'],
                                        mask_zero=True,
                                        name='embed'))(input_block)

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=model_params['bilstm_0_units'], return_sequences=True, name='BiLSTM_0')))(x)
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=model_params['bilstm_1_units'], return_sequences=False, name='BiLSTM_1')))(x)

    x = tf.keras.layers.Dense(units=model_params['dense_0_units'], activation='relu', name='Dense_0')(x)
    x = tf.keras.layers.Dropout(rate=model_params['drop_0_rate'], name='Drop_0')(x)
    x = tf.keras.layers.Dense(units=model_params['dense_1_units'], activation='relu', name='Dense_1')(x)
    x = tf.keras.layers.Dense(units=3, activation='softmax', name='dense_final')(x)
    #
    # # na wyjściu bierzemy tylko element wynik dla elementu środkowegoboczkar
    model = tf.keras.models.Model(inputs=input_block, outputs=x[:, 1])
    #
    model.compile(optimizer=tf.keras.optimizers.Adam(model_params['initial_lr']),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
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


'''@nni.variable(nni.quniform(50, 300, 25), name=embedding_dimension)'''
embedding_dimension = 200
'''@nni.variable(nni.choice(16, 32, 48, 64, 96, 128), name=bilstm_0_units)'''
bilstm_0_units = 128
'''@nni.variable(nni.choice(16, 32, 48, 64, 96, 128), name=bilstm_1_units)'''
bilstm_1_units = 128
'''@nni.variable(nni.quniform(16, 128, 16), name=dense_0_units)'''
dense_0_units = 128
'''@nni.variable(nni.uniform(0., 0.5), name=drop_0_rate)'''
drop_0_rate = 0.5
'''@nni.variable(nni.quniform(16, 128, 16), name=dense_1_units)'''
dense_1_units = 128
'''@nni.variable(nni.uniform(0.05, 0.7), name=lr_reduction_factor)'''
lr_reduction_factor = 0.1
'''@nni.variable(nni.uniform(5e-5, 1e-3), name=initial_lr)'''
initial_lr = 0.0005
'''@nni.variable(nni.choice(16,32,64), name=batch_size)'''
batch_size = 64

model_params = {
    'vocab_size': 8000,
    'output_sequence_length': 45,
    'embedding_dimension': int(embedding_dimension),
    'window_size': 3,

    'bilstm_0_units': int(bilstm_0_units),
    'bilstm_1_units': int(bilstm_0_units),
    'dense_0_units': int(dense_0_units),
    'drop_0_rate': drop_0_rate,
    'dense_1_units': int(dense_1_units),
    'initial_lr': initial_lr,
    'lr_reduction_factor':lr_reduction_factor,
    'batch_size': int(batch_size),
    'max_epochs': 15

}

# add BOM and EOM signals to texts to include them in vocabulary
texts = np.append(texts, [BOM_SIGNAL, EOM_SIGNAL])

vectorizer = tf.keras.layers.TextVectorization(max_tokens=model_params['vocab_size'],
                                               output_sequence_length=model_params['output_sequence_length'],
                                               pad_to_max_tokens=True,
                                               output_mode='int',
                                               name='Vectorizer')
vectorizer.adapt(texts)

model_definition = define_model_bilstm
RANDOM_STATES = [123, 12, 42]
with mlflow.start_run(experiment_id=eid, nested=False, tags={'master': True}) as master_run:
    master_results = {}
    for k in [1]:
        for i in range(0, 3):
            rs = RANDOM_STATES[i]

            train_data, val_data = split_dataset(df, rs)
            train_texts, train_labels = prepare_records(train_data, vectorizer)
            val_texts, val_labels = prepare_records(val_data, vectorizer)

            tf.keras.backend.clear_session()
            model = model_definition(model_params)

            with mlflow.start_run(experiment_id=eid,
                                  run_name=f"{master_run.data.tags['mlflow.runName']}-{k}-{rs}", nested=True,
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
                              tf.keras.callbacks.ReduceLROnPlateau(factor=model_params['lr_reduction_factor'],
                                                                   patience=3,
                                                                   verbose=1),
                              tf.keras.callbacks.EarlyStopping(min_delta=1e-4,
                                                               patience=6,
                                                               restore_best_weights=True),
                              tf.keras.callbacks.EarlyStopping(baseline=0.94,
                                                               monitor='val_accuracy',
                                                               patience=4,
                                                               restore_best_weights=True),
                              tf.keras.callbacks.LambdaCallback(on_epoch_end=report_epoch_progress)
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
                """@nni.report_intermediate_result(master_results[i])"""
    report_parameters(model_params, model, model_definition,
                      master_run)  # model from the last iteration, params are shared among all of them
    master_results_ = report_master_results(master_results)
    master_results_ = {
        'default': master_results_['Accuracy'],
        'loss': master_results_['Loss_eval']
    }
    """@nni.report_final_result(master_results_)"""
