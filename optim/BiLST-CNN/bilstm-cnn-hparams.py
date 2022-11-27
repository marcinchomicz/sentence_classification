import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
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

for m in [pd, tf, tfa, mlflow]:
    print(f"{m.__name__:15s}\t{m.__version__}")

OUTPUTPATH = "/mnt/workdata/_WORK_/mail_zonning/mail_zoning/sandbox/"
DATAPATH = "/mnt/workdata/_WORK_/mail_zonning/mail_zoning/dataset/enron_files_annotated/"
FILESTORE = "/mnt/workdata/_WORK_/mail_zonning/mail_zoning/tmp/"
MLFLOW_DIR = "file:///home/chomima5/mlruns/"


def prepare_dataset(datapath: str):
    def extract_text(text, flags: dict):
        text = text.split('\n')
        idx = 0
        while text[idx][:2] not in flags:
            idx += 1
        labels = [flags[t[:2]] for t in text[idx:] if len(t) > 1]
        text = [t[2:] for t in text[idx:]]
        return text, labels

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
    train_labels = train_data['label'].values
    train_data = train_data['sentence'].values

    test_data = data.loc[data['doc'].isin(test_emails)]
    test_labels = test_data['label'].values
    test_data = test_data['sentence'].values

    return train_data, train_labels, test_data, test_labels


df = prepare_dataset(DATAPATH)
train_data, train_labels, test_data, test_labels = split_dataset(df)

mlflow.set_tracking_uri(MLFLOW_DIR)
ENAME = 'ParamSearch_BiLSTM-CNN'

try:
    eid = mlflow.get_experiment_by_name(ENAME).experiment_id
except:
    eid = mlflow.create_experiment(ENAME)


def define_model_LSTM_CNN(adapted_text_vectorization_layer, model_params: dict, label_count: int):
    inputs = tf.keras.layers.Input(
        shape=(1,),
        dtype=tf.string)
    x = adapted_text_vectorization_layer(inputs)
    x = tf.keras.layers.Embedding(
        input_dim=len(adapted_text_vectorization_layer.get_vocabulary()),
        output_dim=model_params['embedding_dimension'],
        mask_zero=True,
        name='Embedder')(x)
    x = tf.keras.layers.Bidirectional(
        layer=tf.keras.layers.LSTM(
            units=model_params['lstm_units'],
            return_sequences=True),
        name='BiLSTM')(x)
    x = tf.keras.layers.Convolution1D(
        filters=model_params['conv1D_size'],
        kernel_size=model_params['conv1D_kernel_size'],
        strides=1,
        padding='valid',
        activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(
        pool_size=model_params['pool1D_pool_size'], name='maxpool_1')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(
        rate=model_params['drop_0_ratio'],
        name='drop_0')(x)
    x = tf.keras.layers.Dense(
        units=model_params['dense_1_size'],
        activation=model_params['dense_1_activation'],
        name='Dense_1',
        use_bias=True,
        bias_initializer='glorot_uniform')(x)
    x = tf.keras.layers.Dropout(
        rate=model_params['drop_1_ratio'],
        name='drop_1')(x)
    output_layer = tf.keras.layers.Dense(
        units=label_count,
        use_bias=True,
        activation='softmax',
        name='Output')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=output_layer)
    return model


'''@nni.variable(nni.quniform(2000, 8000, 500), name=vocab_size)'''
vocab_size = 7000

'''@nni.variable(nni.quniform(25, 300, 25), name=embedding_dimension)'''
embedding_dimension = 200

'''@nni.variable(nni.choice(16, 32, 48, 64, 96, 128), name=lstm_units)'''
lstm_units = 128

'''@nni.variable(nni.quniform(8, 128, 8), name=dense_1_size)'''
dense_1_size = 64

'''@nni.variable(nni.uniform(0.05, 0.7), name=lr_reduction_factor)'''
lr_reduction_factor = 0.1

'''@nni.variable(nni.uniform(5e-5, 1e-3), name=initial_lr)'''
initial_lr = 0.1

'''@nni.variable(nni.choice(16, 32, 48, 64, 96, 128), name=conv1d_size)'''
conv1d_size = 32

'''@nni.variable(nni.choice(3,4,5), name=conv1d_kernel_size)'''
conv1d_kernel_size = 3

'''@nni.variable(nni.uniform(0., 0.5), name=drop_0_ratio)'''
drop_0_ratio = 0.

'''@nni.variable(nni.uniform(0., 0.5), name=drop_1_ratio)'''
drop_1_ratio = 0.

'''@nni.variable(nni.choice("elu", "relu", "gelu", "selu"), name=dense_1_activation)'''
dense_1_activation = 'relu'

mp_LSTM_CNN = {
    'vocab_size': int(vocab_size),
    'embedding_dimension': int(embedding_dimension),
    'lstm_units': int(lstm_units),
    'dense_1_size': int(dense_1_size),
    'dense_1_activation': dense_1_activation,
    'rnn_variant': 'BiLSTM-CNN',
    'max_epochs': 100,
    'lr_reduction_factor': lr_reduction_factor,
    'conv1D_size': int(conv1d_size),
    'conv1D_kernel_size': int(conv1d_kernel_size),
    'pool1D_pool_size': 2,
    'drop_0_ratio': drop_0_ratio,
    'drop_1_ratio': drop_1_ratio,
    'initial_lr': initial_lr
}

model_params = mp_LSTM_CNN
model_definition = define_model_LSTM_CNN

# create and adapt vectorization layer
vectorization_layer = tf.keras.layers.TextVectorization(max_tokens=model_params['vocab_size'],
                                                        output_sequence_length=45,
                                                        pad_to_max_tokens=True,
                                                        output_mode='int',
                                                        name='Vectorizer')
vectorization_layer.adapt(data=df['sentence'].values)
label_count = df['label'].nunique()

model = model_definition(vectorization_layer, model_params, label_count)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])


def report_epoch_progress(epoch, logs):
    mlflow.log_metrics(logs)


with mlflow.start_run(experiment_id=eid, nested=False, tags={'master': True}) as master_run:
    master_results = {}
    mlflow.log_params(params=model_params)
    mlflow.log_text(json.dumps(json.loads(model.to_json()), indent=3), 'model_schema.txt')
    mlflow.log_text(inspect.getsource(model_definition), 'model_building_method.txt')
    mlflow.log_dict(model_params, 'model_params.txt')
    mlflow.log_dict({'cuda_device': os.environ['CUDA_VISIBLE_DEVICES']}, 'enironment_settings.txt')
    diagram_filename = f"{FILESTORE}/diagram_{master_run.data.tags['mlflow.runName']}.png"
    img = tf.keras.utils.plot_model(model, to_file=diagram_filename, show_shapes=True, show_dtype=True,
                                    show_layer_names=True, show_layer_activations=True)
    mlflow.log_artifact(local_path=diagram_filename)
    os.remove(diagram_filename)

    for i in range(0, 3):
        tf.keras.backend.clear_session()
        with mlflow.start_run(experiment_id=eid,
                              run_name=f"{master_run.data.tags['mlflow.runName']}-{i}", nested=True,
                              tags={'master': False}) as run:
            mlflow.log_params(params=model_params)
            mlflow.log_text(json.dumps(json.loads(model.to_json()), indent=3), 'model_schema')
            mlflow.log_text(inspect.getsource(model_definition), 'model_building_method.txt')
            mlflow.log_dict(model_params, 'model_params.txt')
            mlflow.log_dict({'cuda_device': os.environ['CUDA_VISIBLE_DEVICES']}, 'enironment_settings.txt')
            diagram_filename = f"{FILESTORE}diagram_{run.data.tags['mlflow.runName']}.png"
            img = tf.keras.utils.plot_model(model, to_file=diagram_filename, show_shapes=True, show_dtype=True,
                                            show_layer_names=True, show_layer_activations=True)
            mlflow.log_artifact(local_path=diagram_filename)
            os.remove(diagram_filename)

            model.fit(x=train_data,
                      y=train_labels,
                      epochs=model_params['max_epochs'],
                      validation_data=(test_data, test_labels),
                      callbacks=[
                          tf.keras.callbacks.ReduceLROnPlateau(factor=model_params['lr_reduction_factor'],
                                                               patience=5,
                                                               verbose=1),
                          tf.keras.callbacks.EarlyStopping(min_delta=1e-4,
                                                           patience=10,
                                                           restore_best_weights=True),
                          tf.keras.callbacks.LambdaCallback(on_epoch_end=report_epoch_progress)
                      ])
            eval_loss, eval_acc = model.evaluate(test_data, test_labels)
            y_pred_ = model.predict(test_data)
            y_true = test_labels
            y_pred = np.argmax(y_pred_, axis=1)
            final_metrics = {
                'Loss_eval': eval_loss,
                'Accuracy': accuracy_score(y_true, y_pred),
                'Precision': precision_score(y_true, y_pred, average='weighted'),
                'Recall': recall_score(y_true, y_pred, average='weighted'),
                'F1-score': f1_score(y_true, y_pred, average='weighted')
            }
            mlflow.log_metrics(final_metrics)
            mlflow.log_dict(classification_report(y_true, y_pred, output_dict=True), 'classification_report.txt')
            master_results[i] = final_metrics
            final_metrics_ = {
                'Loss_eval': eval_loss,
                'default': accuracy_score(y_true, y_pred),
            }
            """@nni.report_intermediate_result(final_metrics)"""

    mlflow.log_dict(master_results, 'iteration_results.txt')
    master_results_ = pd.DataFrame(master_results).mean(axis=1).to_dict()
    mean_vals = pd.DataFrame(master_results).mean(axis=1)
    std_vals = pd.DataFrame(master_results).std(axis=1)
    mean_vals.index = [f"{c}_mean" for c in mean_vals.index]
    std_vals.index = [f"{c}_std" for c in std_vals.index]
    _ = {**mean_vals.to_dict(), **std_vals.to_dict()}
    mlflow.log_metrics(_)
    master_results_ = {
        'default': master_results_['Accuracy'],
        'loss': master_results_['Loss_eval']
    }
    """@nni.report_final_result(master_results_)"""
