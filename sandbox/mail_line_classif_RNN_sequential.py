
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
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

DATAPATH = "/mnt/workdata/_WORK_/mail_zonning/mail_zoning/dataset/enron_files_annotated/"

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

mp = {
    'vocab_size': 7500,
    'hidden_size': 32,
    'lr_reduction_factor': 0.1,
    'rnn_variant': 'BiLSTM',
}



mlflow.set_tracking_uri("file:///home/chomima5/mlruns")
ENAME = 'SencenceClassifCNN'

try:
    eid = mlflow.get_experiment_by_name(ENAME).experiment_id
except:
    eid = mlflow.create_experiment(ENAME)


def define_model_BiLSTM(adapted_text_vectorization_layer, model_params: dict, label_count: int):
    """ Define model using functional API"""
    tf.keras.backend.clear_session()

    # Input layer is required, datatype and shape must be provided as well
    inputs = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
    x = adapted_text_vectorization_layer(inputs)

    # define embedding layer
    # get real vocab size, after adapting TextVectorization
    x = tf.keras.layers.Embedding(input_dim=len(adapted_text_vectorization_layer.get_vocabulary()),
                                           output_dim=model_params['hidden_size'],
                                           mask_zero=True, name='Embedder')(x)
    #define BiLSTM and Dense layers
    x = tf.keras.layers.Bidirectional(
        layer=tf.keras.layers.LSTM(units=mp['hidden_size']), name='BiLSTM')(x)
    x = tf.keras.layers.Dense(units=mp['hidden_size'], activation='relu', name='Dense_1')(x)
    output_layer = tf.keras.layers.Dense(units=label_count, use_bias=True, activation='softmax', name='Output')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=output_layer)
    return model

def define_model_CNN(adapted_text_vectorization_layer, model_params: dict, label_count:int):
    tf.keras.backend.clear_session()
    inputs=tf.keras.layers.Input(shape=(1,), dtype=tf.string)
    x=adapted_text_vectorization_layer(inputs)
    x = tf.keras.layers.Embedding(input_dim=len(adapted_text_vectorization_layer.get_vocabulary()),
                                  output_dim=model_params['hidden_size'],
                                  mask_zero=True,
                                  name='Embedder')(x)
    x1= tf.keras.layers.Convolution1D(
        filters=model_params['Conv1D_size'],
        kernel_size=model_params['Conv1D_kernel_size'],
        strides=1,
        padding='valid',
        activation='relu')(x)

    x1 = tf.keras.layers.MaxPooling1D(pool_size=model_params['Pool1D_pool_size'])(x1)
    x1 = tf.keras.layers.Flatten()(x1)

    x= tf.keras.layers.Dense(model_params['hidden_size'])(x1)
    output_layer=tf.keras.layers.Dense(units=label_count, use_bias=True, activation='softmax', name='Output')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=output_layer)
    return model

def define_model_CNN_LSTM(adapted_text_vectorization_layer, model_params: dict, label_count:int):
    tf.keras.backend.clear_session()
    inputs=tf.keras.layers.Input(shape=(1,), dtype=tf.string)
    x=adapted_text_vectorization_layer(inputs)
    x = tf.keras.layers.Embedding(input_dim=len(adapted_text_vectorization_layer.get_vocabulary()),
                                  output_dim=model_params['hidden_size'],
                                  mask_zero=True,name='Embedder')(x)


    x1= tf.keras.layers.Convolution1D(
        filters=model_params['Conv1D_size'],
        kernel_size=model_params['Conv1D_kernel_size'],
        strides=1,
        padding='valid',
        activation='relu')(x)
    x1 = tf.keras.layers.MaxPooling1D(pool_size=model_params['Pool1D_pool_size'])(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Dense(model_params['hidden_size'])(x1)

    x2 = tf.keras.layers.Bidirectional(
        layer=tf.keras.layers.LSTM(units=mp['hidden_size'], return_sequences=True), name='BiLSTM')(x)
    x2 = tf.keras.layers.Dense(units=mp['hidden_size'],
                                    activation='relu',
                                    name='Dense_1',
                                    use_bias=True,
                                    bias_initializer='glorot_uniform')(x2)

    x=tf.keras.layers.Concatenate(axis=1)([x1, x2])


    output_layer=tf.keras.layers.Dense(units=label_count, use_bias=True, activation='softmax', name='Output')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=output_layer)
    return model


vocab_size_scope = [7500] # 1000, 2500, 5000, 7500, 10000
hidden_size_scope = [64] # 16, 32, 64, 96, 128, 256
rnn_variants = ['CNN'] # 'BiLSTM', 'StackedBiLSTM', 'BiGRU', 'StackedBiGRU'

scope = list(itertools.product(vocab_size_scope, hidden_size_scope, rnn_variants))
for vocab_size, hidden_size, variant in tqdm(scope, total=len(scope)):
    mp['vocab_size'] = vocab_size
    mp['hidden_size'] = hidden_size
    mp['rnn_variant'] = variant
    mp['split'] = 'document_based'
    mp['output_sequence_length'] = 300
    mp['Conv1D_size'] = 64
    mp['Conv1D_kernel_size']=3
    mp['Pool1D_pool_size']=2


    # create and adapt vectorization layer
    vectorization_layer = tf.keras.layers.TextVectorization(max_tokens=mp['vocab_size'],
                                                            output_sequence_length=300,
                                                            pad_to_max_tokens=True,
                                                            output_mode='int',
                                                            name='Vectorizer')
    vectorization_layer.adapt(data=df['sentence'].values)
    label_count = df['label'].nunique()


    model=define_model_BiLSTM(vectorization_layer,mp,label_count)

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
