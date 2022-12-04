import tensorflow as tf
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view



class ErrorWindowSizeToSmall(Exception):
    """ Exception raised when window size is smaller than 3 """
    def __init__(self):
        message = 'Window size must be greater or equal 3.'
        super().__init__(message)


class ErrorWindowSizeMustBeOdd(Exception):
    """ Exception raised when window size is even """
    def __init__(self):
        message = 'Windows size must be odd number.'
        super().__init__(message)

class ErrorParamNotFound(Exception):
    """ Exception raised when at least one of expected parameters was not found """
    def __init__(self, message):
        super().__init__(message)


class MovingWindowSentenceClassifier:
    # noinspection SpellCheckingInspection
    """
        Sentence-level classifier with sliding window approach.

        There are two main components:
        - Vectorizer - text vectorization layer
        - Model - multiclass classification model based on two convolutional and two
        Bidirectional GRU layers,separated with dropout.

        The class exposes the interface of tf.keras.models.Mode as itself,
        so any operation possible on Model instance are available for the
        class instance.

        The structure of model is as follows:
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

    # parameters that must be provided
    # noinspection SpellCheckingInspection
    MODEL_PARAMS_EXPECTED_KEYS = {
        'output_class_count',
        'vocab_size',
        'output_sequence_length',
        'embedding_dimension',
        'window_size',
        'conv1d_0_units', 'conv1d_0_kernelsize', 'conv1d_0_padding', 'conv1d_0_activation',
        'conv1d_1_units', 'conv1d_1_kernelsize', 'conv1d_1_padding', 'conv1d_1_activation',
        'gru_0_units', 'gru_1_units', 'drop_0_rate'
    }

    def __init__(self,
                 model_params: dict,
                 bod_line: str,
                 eod_line: str,
                 corpus ,
                 **kwargs):
        """
        Instantiates classifier object, based on configuration in model_params.
        The dictionary must adhere to the following structure:

            {

            output_class_count: int - the number of output classes,
            vocab_size: int - the number of vocabulary items for text vectorization,
            output_sequence_length: int -  the size of vectorized sequence,
            embedding_dimension: int - the length of embedding vector,
            window_size: int - sliding window size, must be odd number greater or equal 3,
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
            'drop_0_rate': float - dropout rate between CNN and RNN components
            }

        :param model_params: a dictionary with params for model, see description above
        :param bod_line: specific line, added at the begin of each document
        :param eod_line: specific line, added at the end of each document
        :param corpus: list of strings used to feed text vectorization, typically all documents from train set
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
        :param model_params: model_params, dictionary, see __init__ documentation
        :return: model_params dictionary after validation
        """
        lacking_params=self.MODEL_PARAMS_EXPECTED_KEYS.difference(model_params)
        if lacking_params!=set():
            raise ErrorParamNotFound(f"Params: {lacking_params} not found")
        if model_params['window_size'] < 3:
            raise ErrorWindowSizeToSmall
        if model_params['window_size'] % 2 == 0:
            raise ErrorWindowSizeMustBeOdd
        return model_params

    def _prepare_records(self,
                         data: pd.DataFrame,
                         doc_id: str = 'doc',
                         line_index: str = 'idx',
                         line: str = 'sentence',
                         label: str = 'label') -> (np.ndarray, np.ndarray) :
        """
        Prepare windowed records for classification.
        Each record is composed of <window_size> lines of text

        :param data: pandas dataframe containing documents
        :param doc_id: name of column with document id
        :param line_index: name of column with index of line within the document
        :param line: name of column with line of the document
        :param label: name of column with label for the line
        :return: array of windowed blocks, array of labels for blocks
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
                        doc_id: str = 'doc',
                        line_index: str = 'idx',
                        line: str = 'sentence',
                        label: str = 'label',
                        ):
        """
        Prepare windowed records for classification.
        Each record is composed of <window_size> lines of text
        The records are stored internally as train records and labels.

        :param data: pandas dataframe containing documents
        :param doc_id: name of column with document id
        :param line_index: name of column with index of line within the document
        :param line: name of column with line of the document
        :param label: name of column with label for the line

        """
        self.train_texts, self.train_labels = self._prepare_records(data, doc_id, line_index, line, label)

    def prepare_validation_records(self,
                                   data: pd.DataFrame,
                                   doc_id: str = 'doc',
                                   line_index: str = 'idx',
                                   line: str = 'sentence',
                                   label: str = 'label'):
        """
        Prepare windowed records for classification.
        Each record is composed of <window_size> lines of text
        The records are stored internally as validation records and labels.

        :param data: pandas dataframe containing documents
        :param doc_id: name of column with document id
        :param line_index: name of column with index of line within the document
        :param line: name of column with line of the document
        :param label: name of column with label for the line
        """
        self.validation_texts, self.validation_labels = self._prepare_records(data, doc_id, line_index, line, label)

    def prepare_test_records(self,
                             data: pd.DataFrame,
                             doc_id: str = 'doc',
                             line_index: str = 'idx',
                             line: str = 'sentence',
                             label: str = 'label'):
        """
        Prepare windowed records for classification.
        Each record is composed of <window_size> lines of text
        The records are stored internally as test records and labels.

        :param data: pandas dataframe containing documents
        :param doc_id: name of column with document id
        :param line_index: name of column with index of line within the document
        :param line: name of column with line of the document
        :param label: name of column with label for the line
        """
        self.test_texts, self.test_labels = self._prepare_records(data, doc_id, line_index, line, label)

    def get_model(self)-> tf.keras.models.Model:
        """
        Create classifier.

        :return: the instance of tf.keras.models.Model, must be compiled before use.
        """
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
        Enable access to attribute/method of self.model
        Any method of tf.keras.models.Model instantiated as self.model can be used transparently on the class instance
        """
        return getattr(self.model, name)



