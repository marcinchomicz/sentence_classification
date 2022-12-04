import tensorflow as tf
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class ErrorWindowSizeToSmall(Exception):
    """ Exception raised when window size is smaller than 3 """
    def __init__(self):
        MESSAGE = 'Window size must be greater or equal 3.'
        super().__init__(MESSAGE)


class ErrorWindowSizeMustBeOdd(Exception):
    """ Exception raised when window size is even """
    def __init__(self):
        MESSAGE = 'Windows size must be odd number.'
        super().__init__(MESSAGE)

class ErrorParamNotFound(Exception):
    """ Exception raised when at least one of expected paramweters was not found """
    def __init__(self, message):
        super().__init__(message)


class MovingWindowSentenceClassifier:
    """
    Sentence-level classifier with sliding window approach.

    There are two main components:
    - Vectorizer - text vectorization layer
    - Model - multiclass classification model based on two convolutional and two
    Bidirectional GRU layers,separated with dropout.

    The class exposes the interface of tf.keras.models.Mode as itself,
    so any operation possible on Model instance are available for the
    class instance.

    The structure of model is as follow:
    - Input
    -- Embedding layer
    -- BatchNormalization
    -- TimDistributed
    --- Conv1D
    -- TimDistributed
    --- Conv1D
    -- Dropout
    -- TimeDistributed
    --- Flatten
    -- Bidirectional
    --- GRU (return sequence=True)
    -- Bidirectional
    --- GRU (return_sequence=False)
    - Dense (output)

    The response variable should be provided as sparse value,
    and sparse categorical crossentropy should be used.

    """

    def __init__(self,
                 model_params: dict,
                 bod_line: str,
                 eod_line: str,
                 corpus ,
                 **kwargs):
        """
        Instantiates classifier object, based on configuration in model_params.
        The dictionary must adhere to the dollowing structure:

            {

            output_class_count: int - the number of output classes,
            vocab_size: int - the number of vocabulary items for text vectorization,
            output_sequence_length: int -  the size of vectorized sequence,
            embedding_dimension: int - the length of embedding vector,
            window_size: int - sliding window size, must be odd number greater or equel 3,
            'conv1d_0_units': int - number of units in first convolution,
            'conv1d_0_kernelsize': int - kernelsize for first convolution,
            'conv1d_0_padding': str - padding mode for first convolution,
            'conv1d_0_activation': str - activation function for first convolution,
            'conv1d_1_units': int - number of units in second convolution,
            'conv1d_1_kernelsize': int - kernelsize for second convolution,
            'conv1d_1_padding': str - padding mode for second convolution,
            'conv1d_1_activation': str - activation function for second convolution,
            'gru_0_units': int - number of units in first GRU layer,
            'gru_1_units': int - number of units in second GRU layer,,
            'drop_0_rate': float - dropoute rate between CNN and RNN components
            }

        :param model_params: a dictionary with params for model, see description above
        :param bod_line: specific line, added at the begin of each document
        :param eod_line: specific line, added at the end of each document
        :param corpus: list of strings used to feed text vectorization, typically all docuemnts from train set
        :param kwargs: other parameters, passed directly to Vectorizer
        """
        self._params = self._set_params(model_params)
        self.bod_line = bod_line
        self.eod_line = eod_line
        self.vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=self._params['vocab_size'],
            output_sequence_length=self._params['output_sequence_length'],
            output_mode='int',
            name='Vectorizer', **kwargs)
        self.vectorizer.adapt(corpus)
        self.model = self.get_model()
        self.train_texts = np.ndarray([])
        self.train_labels = np.ndarray([])
        self.validation_texts = np.ndarray([])
        self.validation_labels = np.ndarray([])
        self.test_texts = np.ndarray([])
        self.test_labels = np.ndarray([])


    def _set_params(self, model_params: dict):
        """
        Perform validation of model_params dictionary structure
        :param model_params: model_params, dicionary, see __init__ documentation
        :return: model_params dictionary after validation
        """
        EXPECTED={
            'output_class_count',
            'vocab_size',
            'output_sequence_length',
            'embedding_dimension',
            'window_size',
            'conv1d_0_units', 'conv1d_0_kernelsize', 'conv1d_0_padding', 'conv1d_0_activation',
            'conv1d_1_units', 'conv1d_1_kernelsize', 'conv1d_1_padding', 'conv1d_1_activation',
            'gru_0_units', 'gru_1_units', 'drop_0_rate'
        }
        lacking_params=EXPECTED.difference(model_params)
        if lacking_params!=set():
            raise ErrorParamNotFound(f"Params: {lacking_params} not found")
        if model_params['window_size'] < 3:
            raise ErrorWindowSizeToSmall
        if model_params['window_size'] % 2 == 0:
            raise ErrorWindowSizeMustBeOdd
        return model_params

    def _prepare_records(self, data: pd.DataFrame,
                         doc_id: str = 'doc', line_index: str = 'idx', line: str = 'sentence', label: str = 'label'):
        """
        Prepare windowed records for classification.
        Each record is composed of <windo_size> lines of text

        :param data:
        :param doc_id:
        :param line_index:
        :param line:
        :param label:
        :return:
        """

        data = data[[doc_id, line_index, line, label]].values
        texts = []
        labels_ = []
        for doc in np.unique(data[:, 0]):
            sents = np.concatenate([np.array([self.bod_line for i in range(self._params['window_size'] // 2)]),
                                    data[data[:, 0] == doc, 2],
                                    np.array([self.eod_line for i in range(self._params['window_size'] // 2)])])
            sents = self.vectorizer(sents).numpy()
            labels = data[data[:, 0] == doc, 3]
            sents = sliding_window_view(
                x=sents,
                window_shape=(self._params['window_size'],
                              self._params['output_sequence_length']))
            for sent in sents:
                texts.append(sent)
            labels_.extend(labels)
        return np.squeeze(texts), np.stack(labels_)

    def prepare_train_records(self, data: pd.DataFrame,
                              doc_id: str = 'doc', line_index: str = 'idx', line: str = 'sentence',
                              label: str = 'label'):
        self.train_texts, self.train_labels = self._prepare_records(data, doc_id, line_index, line, label)

    def prepare_validation_records(self, data: pd.DataFrame,
                                   doc_id: str = 'doc', line_index: str = 'idx', line: str = 'sentence',
                                   label: str = 'label'):
        self.validation_texts, self.validation_labels = self._prepare_records(data, doc_id, line_index, line, label)

    def prepare_test_records(self, data: pd.DataFrame,
                             doc_id: str = 'doc', line_index: str = 'idx', line: str = 'sentence',
                             label: str = 'label'):
        self.test_texts, self.test_labels = self._prepare_records(data, doc_id, line_index, line, label)

    def get_model(self):
        input_block = tf.keras.layers.Input(
            shape=(self._params['window_size'],
                   self._params['output_sequence_length']),
            dtype=tf.int32,
            name='input_block')

        x = tf.keras.layers.Embedding(input_dim=len(self.vectorizer.get_vocabulary()),
                                      output_dim=self._params['embedding_dimension'],
                                      input_length=self._params['output_sequence_length'],
                                      mask_zero=False,
                                      name='embed')(input_block)

        x = tf.keras.layers.BatchNormalization(name='embed_batch_norm')(x)

        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv1D(
                filters=self._params['conv1d_0_units'],
                kernel_size=self._params['conv1d_0_kernelsize'],
                padding=self._params['conv1d_0_padding'],
                activation=self._params['conv1d_0_activation'],
                name='conv1d_0'))(x)

        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv1D(
                filters=self._params['conv1d_1_units'],
                kernel_size=self._params['conv1d_1_kernelsize'],
                padding=self._params['conv1d_1_padding'],
                activation=self._params['conv1d_1_activation'],
                name='conv1d_1'))(x)

        x = tf.keras.layers.Dropout(rate=self._params['drop_0_rate'])(x)

        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)

        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(self._params['gru_0_units'], return_sequences=True))(x)

        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(self._params['gru_1_units'], return_sequences=False))(x)

        x = tf.keras.layers.Dense(
            units=self._params['output_class_count'],
            activation='softmax')(x)
        return tf.keras.models.Model(inputs=input_block, outputs=x)

    def __getattr__(self, name):
        """
        This method enables to access an attribute/method of self.model.
        Thus, any method of keras.Model() can be used transparently from a SubModel object
        """
        return getattr(self.model, name)


if __name__ == "__main__":
    import os
    from sklearn.model_selection import KFold

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
