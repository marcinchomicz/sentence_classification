import os

os.environ['CUDA_VISIBLE_DEVICES'] = ""
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import mlflow
import json
from sklearn.model_selection import train_test_split
import itertools
from tqdm import tqdm

for m in [pd, tf, tfa, mlflow]:
    print(f"{m.__name__:15s}\t{m.__version__}")

OUTPUTPATH = "/mnt/workdata/_WORK_/mail_zonning/mail_zoning/sandbox"

DATAPATH = "/mnt/workdata/_WORK_/mail_zonning/repos/Quagga/Datasets/Enron/annotated_lines/"

def prepare_dataset(datapath:str):

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

df=prepare_dataset(DATAPATH)
train_data, train_labels, test_data, test_labels = split_dataset(df)
print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)


mp = {
    'vocab_size': 10000,
    'hidden_size': 32,
    'lr_reduction_factor': 0.1,
    'rnn_variant': 'BiLSTM',
}


def define_layers(model_params: dict, vectorization_layer):
    if model_params['rnn_variant'] == 'StackedBiLSTM':
        rnn_layers = [
            tf.keras.layers.Bidirectional(
                layer=tf.keras.layers.LSTM(units=model_params['hidden_size'], return_sequences=True), name='BiLSTM_0'),
            tf.keras.layers.Bidirectional(
                layer=tf.keras.layers.LSTM(units=model_params['hidden_size']), name='BiLSTM_1')
        ]
    elif model_params['rnn_variant'] == 'BiGRU':
        rnn_layers = [
            tf.keras.layers.Bidirectional(
                layer=tf.keras.layers.GRU(units=model_params['hidden_size'], return_sequences=False, name='BiGRU'))
        ]
    elif model_params['rnn_variant'] == 'StackedBiGRU':
        rnn_layers = [
            tf.keras.layers.Bidirectional(
                layer=tf.keras.layers.GRU(units=model_params['hidden_size'], return_sequences=True, name='BiGRU_0')),
            tf.keras.layers.Bidirectional(
                layer=tf.keras.layers.GRU(units=model_params['hidden_size'], return_sequences=False, name='BiGRU_1'))
        ]
    else:
        rnn_layers = [tf.keras.layers.Bidirectional(
            layer=tf.keras.layers.LSTM(units=model_params['hidden_size']), name='BiLSTM')]

    layers = [
        vectorization_layer,
        tf.keras.layers.Embedding(input_dim=len(vect_layer.get_vocabulary()),
                                  input_length=model_params['output_sequence_length'],
                                  output_dim=model_params['hidden_size'],
                                  mask_zero=True,
                                  name='Embedder')
    ]
    layers.extend(rnn_layers)
    layers.append(tf.keras.layers.Dense(units=model_params['hidden_size'], activation='relu', name='Dense_1'))
    layers.append(tf.keras.layers.Dense(units=label_count, use_bias=True, activation='softmax', name='Output'))

    return layers

mlflow.set_tracking_uri("file:///home/chomima5/mlruns")
ENAME = 'SencenceClassifRNN_Sequential'

try:
    eid = mlflow.get_experiment_by_name(ENAME).experiment_id
except:
    eid = mlflow.create_experiment(ENAME)


vocab_size_scope = [2000] # 1000, 2500, 5000, 7500, 10000
hidden_size_scope = [64] # 16, 32, 64, 96, 128, 256
rnn_variants = ['BiLSTM'] # 'BiLSTM', 'StackedBiLSTM', 'BiGRU', 'StackedBiGRU'

scope = list(itertools.product(vocab_size_scope, hidden_size_scope, rnn_variants))

for vocab_size, hidden_size, variant in tqdm(scope, total=len(scope)):
    mp['vocab_size'] = vocab_size
    mp['hidden_size'] = hidden_size
    mp['rnn_variant'] = variant
    mp['split'] = 'document_based'
    mp['output_sequence_length'] = 300

    tf.keras.backend.clear_session()
    vect_layer = tf.keras.layers.TextVectorization(max_tokens=mp['vocab_size'],
                                                   output_mode='int',
                                                   name='Vectorizer')
    vect_layer.adapt(data=df['sentence'].values)
    label_count = df['label'].nunique()

    inputs=tf.keras.layers.Input(shape=(1,), dtype=tf.string)
    vectors=vect_layer(inputs)
    embeddings = tf.keras.layers.Embedding(
        input_dim=len(vect_layer.get_vocabulary()), input_length=mp['output_sequence_length'],
        output_dim=mp['hidden_size'], mask_zero=True, name='Embedder')(vectors)
    bilstm = tf.keras.layers.Bidirectional(
        layer=tf.keras.layers.LSTM(units=mp['hidden_size']), name='BiLSTM')(embeddings)
    dense_1 = tf.keras.layers.Dense(units=mp['hidden_size'], activation='relu', name='Dense_1')(bilstm)
    output_layer=tf.keras.layers.Dense(units=label_count, use_bias=True, activation='softmax', name='Output')(dense_1)

    model = tf.keras.models.Model(inputs=inputs, outputs=output_layer)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    with mlflow.start_run(experiment_id=eid, tags={'model_type': 'SimpleRNN'}) as run:
        mlflow.log_params(params=mp)
        mlflow.log_text(json.dumps(json.loads(model.to_json()), indent=3), 'model_schema')
        model.fit(x=train_data,
                  y=train_labels,
                  epochs=100,
                  validation_data=(test_data, test_labels),
                  callbacks=[
                      tf.keras.callbacks.ReduceLROnPlateau(factor=mp['lr_reduction_factor'],
                                                           patience=5,
                                                           verbose=1),
                      tf.keras.callbacks.EarlyStopping(min_delta=1e-4,
                                                       patience=10,
                                                       restore_best_weights=True)])
        eval_loss, eval_acc = model.evaluate(test_data, test_labels)
        mlflow.log_metrics({'acc_eval': eval_acc, 'loss_eval': eval_loss})
