import tensorflow as tf
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from abc import ABC,  abstractmethod


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


class BaseSentenceClassifier(ABC):

    def __init__(self, model_params: dict, bod_line: str, eod_line: str, corpus, **kwargs):
        self._params = self._set_params(model_params)
        self.bod_line = bod_line
        self.eod_line = eod_line
        self.vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=self._params['vocab_size'],
            output_sequence_length=self._params['output_sequence_length'],
            output_mode='int', name='Vectorizer', **kwargs)
        self.vectorizer.adapt(corpus)
        self.train_texts = np.ndarray([])
        self.train_labels = np.ndarray([])
        self.validation_texts = np.ndarray([])
        self.validation_labels = np.ndarray([])
        self.test_texts = np.ndarray([])
        self.test_labels = np.ndarray([])

    def _set_params(self, model_params: dict):
        return model_params

    @abstractmethod
    def _prepare_records(self, data, doc_id, line_index, line, label):
        pass

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


class MovingWindowSentenceClassifier(BaseSentenceClassifier):
    # noinspection SpellCheckingInspection
    """
        Sentence-level classifier with sliding window approach.

        There are two main components:
        - Vectorizer - text vectorization layer
        - Model - multiclass classification model based on two convolutional and two
        Bidirectional GRU layers,separated with dropout.

        The class exposes the interface of tf.keras.models.Mode as itself,
        so any operation possible on Model instance is available for the
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
                 corpus,
                 bod_line: str,
                 eod_line: str,
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
        super().__init__(model_params=model_params, bod_line=bod_line, eod_line=eod_line, corpus=corpus, **kwargs)
        self.model = self.get_model()

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


class ContextBranchSentenceClassifier(BaseSentenceClassifier):

    """
        Sentence-level classifier with multiple input approach.
        The model has several inputs, with one in the middle for analysed sentence
        and additional inputs for preceeding and following sentences

        There are two main components:
        - Vectorizer - text vectorization layer
        - Model - multiclass classification model based on convolutional and
        Bidirectional LSTM layers, separated with dropout.

        The class exposes the interface of tf.keras.models.Mode as itself,
        so any operation possible on Model instance is available for the
        class instance.

        The structure of model is as follows:

        branches for preceding and following sentences:

        Input
        - TextVectorization
        - Embedding layer
        - BatchNormalization
        - Dense
        - Flatten
        - Drouput
        - Dense

        branch for main sentence:

        Input
        - TextVectorization
        - Embedding layer
        - BatchNormalization
        - Bidirectional
        -- LSTM
        - Conv1D
        - MaxPooling1D
        - Flatten
        - Dense

        Common layers agregatting the pilars above:
        - Concatenate
        - Dropout
        - Dense
        Output (dense)


        The response variable should be provided as sparse value,
        and sparse categorical crossentropy should be used.

        """

    MODEL_PARAMS_EXPECTED_KEYS = {
        "vocab_size",
        "output_sequence_length",
        "context_lines",
        "embedding_dimension",

        "dense_0_pred_size",
        "dense_0_pred_activation",
        "drop_pred_rate",
        "dense_1_pred_size",
        "dense_1_pred_activation",

        "dense_0_post_size",
        "dense_0_post_activation",
        "drop_post_rate",
        "dense_1_post_size",
        "dense_1_post_activation",

        "bilstm_sent_units",
        "conv1D_size",
        "conv1D_kernel_size",
        "pool1d_pool_size",
        "dense_sent_size",
        "dense_sent_activation",
        "merger_dropout_rate",
        "dense_merger_size",
        "dense_merger_activation"
    }

    def __init__(self,
                 model_params: dict,
                 corpus,
                 bod_line: str,
                 eod_line: str,
                 **kwargs):
        """
        Instantiates classifier object, based on configuration in model_params.
        The dictionary must adhere to the following structure:

            {

            output_class_count: int - the number of output classes,
            vocab_size: int - the number of vocabulary items for text vectorization,
            output_sequence_length: int -  the size of vectorized sequence,
            context lines : int - the number of lines considered as context, currently allowed only 3
            embedding_dimension: int - the length of embedding vector,

            dense_0_pred_size,
            dense_0_pred_activation,
            drop_pred_rate,
            dense_1_pred_size,
            dense_1_pred_activation,

            dense_0_post_size,
            dense_0_post_activation,
            drop_post_rate,
            dense_1_post_size,
            dense_1_post_activation,

            bilstm_sent_units,
            conv1D_size,
            conv1D_kernel_size,
            pool1d_pool_size,
            dense_sent_size,
            dense_sent_activation,
            merger_dropout_rate,
            dense_merger_size,
            dense_merger_activation
            }

        :param model_params: a dictionary with params for model, see description above
        :param corpus: list of strings used to feed text vectorization, typically all documents from train set
        :param bod_line: specific line, added at the begin of each document
        :param eod_line: specific line, added at the end of each document
        :param kwargs: other parameters, passed directly to Vectorizer
        """
        super().__init__(model_params=model_params, corpus=corpus, bod_line=bod_line, eod_line=eod_line, **kwargs)
        self.model = self.get_model()

    def _set_params(self, model_params: dict):
        """
        Perform validation of model_params dictionary structure
        :param model_params: model_params, dictionary, see __init__ documentation
        :return: model_params dictionary after validation
        """
        lacking_params=self.MODEL_PARAMS_EXPECTED_KEYS.difference(model_params)
        if lacking_params!=set():
            raise ErrorParamNotFound(f"Params: {lacking_params} not found")
        return model_params


    def get_model(self):
        """
        Create classifier.

        :return: the instance of tf.keras.models.Model, must be compiled before use.
        """
        inputs=[]
        pillars={}
        # Input, vectorization and embedding for preceeding, sentence and following
        for name in ['pred','sent', 'post']:
            input = tf.keras.Input(shape=(1,), dtype=tf.string, name=f'input_{name}')
            inputs.append(input)
            x = self.vectorizer(input)
            x = tf.keras.layers.Embedding(
                input_dim=len(self.vectorizer.get_vocabulary()),
                output_dim=self._params['embedding_dimension'],
                mask_zero=True, name=f'embed_{name}')(x)
            x = tf.keras.layers.BatchNormalization(name=f'norm_{name}')(x)
            pillars[name]=x

        # pillars for preceding and following
        for name in ['pred','post']:
            x = tf.keras.layers.Dense(
                units=self._params[f'dense_0_{name}_size'],
                activation=self._params[f'dense_0_{name}_activation'], name=f'dense_0_{name}')(pillars[name])
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dropout(rate=self._params[f'drop_{name}_rate'], name=f'drop_{name}')(x)
            x = tf.keras.layers.Dense(
                units=self._params[f'dense_1_{name}_size'],
                activation=self._params[f'dense_1_{name}_activation'], name=f'dense_1_{name}')(x)
            pillars[name]=x

        # pillar for current sentence
        x = tf.keras.layers.Bidirectional(
            layer=tf.keras.layers.LSTM(
                units=self._params['bilstm_sent_units'], return_sequences=True), name='bilstm_sent')(pillars['sent'])
        x = tf.keras.layers.Convolution1D(
            filters=self._params['conv1D_size'],
            kernel_size=self._params['conv1D_kernel_size'],
            strides=1, padding='valid', activation='relu', name='conv1d_sent')(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=self._params['pool1d_pool_size'], name='maxpool_sent')(x)
        x = tf.keras.layers.Flatten(name='flatten_sent')(x)
        x = tf.keras.layers.Dense(
            units=self._params['dense_sent_size'],
            activation=self._params['dense_sent_activation'],
            name='dense_sent')(x)
        pillars['sent']=x

        # merging data from preceding, current and following
        merged = tf.keras.layers.Concatenate(axis=1, name='merger')([pillars['pred'], pillars['sent'], pillars['post']])
        merged = tf.keras.layers.Dropout(rate=self._params['merger_dropout_rate'], name='merger_dropout')(merged)
        merged = tf.keras.layers.Dense(
            units=self._params['dense_merger_size'],
            activation=self._params['dense_merger_activation'],
            name='merged_dense')(merged)
        output_layer = tf.keras.layers.Dense(
            units=self._params['output_class_count'],
            use_bias=True,
            activation='softmax',
            name='Output')(merged)
        return tf.keras.models.Model(inputs=inputs, outputs=output_layer)


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
        :return: array of [predecessor, sentence, follower], array of labels
        """

        df=data[[doc_id, line_index, line, label]]
        df['pred']=''
        df['post']=''
        df=df[[doc_id, line_index, 'pred', line, 'post', label]]
        for doc in data[doc_id].unique():
            _ = df.loc[df[doc_id]==doc, line]
            df.loc[df[doc_id]==doc, 'pred'] = _.shift(periods=1, fill_value=self.bod_line)
            df.loc[df[doc_id] == doc, 'post'] = _.shift(periods=-1, fill_value=self.eod_line)
        return df[['pred', line, 'post']].values, df[label].values

    def __getattr__(self, name):
        """
        Enable access to attribute/method of self.model
        Any method of tf.keras.models.Model instantiated as self.model can be used transparently on the class instance
        """
        return getattr(self.model, name)