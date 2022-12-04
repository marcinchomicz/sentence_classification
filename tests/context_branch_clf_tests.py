import unittest

import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sentence_classifier import ContextBranchSentenceClassifier

DATAPATH = "/mnt/workdata/_WORK_/mail_zonning/mail_zoning/dataset/enron_files_annotated/"

BOM_SIGNAL = 'the start of the email signal, no lines before'
EOM_SIGNAL = 'the end of the email signal, no lines after'


class COntextBranchSentenceClassifierTests(unittest.TestCase):

    def prepare_dataset(self, datapath: str):
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

    def split_dataset(self, data: pd.DataFrame, random_state: int):
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

    def test_quick_overall(self):

        model_params = {
            "vocab_size": 8000,
            "output_sequence_length": 45,
            "context_lines": 1,
            "output_class_count": 3,
            "embedding_dimension": 250,
            "dense_0_pred_size": 32,
            "dense_0_pred_activation": "relu",
            "drop_pred_rate": 0.42512612337709293,
            "dense_1_pred_size": 112,
            "dense_1_pred_activation": "relu",

            "dense_0_post_size": 96,
            "dense_0_post_activation": "relu",
            "drop_post_rate": 0.28735873296021,
            "dense_1_post_size": 64,
            "dense_1_post_activation": "relu",

            "bilstm_sent_units": 32,
            "conv1D_size": 48,
            "conv1D_kernel_size": 3,
            "pool1d_pool_size": 2,
            "dense_sent_size": 32,
            "dense_sent_activation": "relu",
            "merger_dropout_rate": 0.14874339700243905,
            "dense_merger_size": 32,
            "dense_merger_activation": "relu",
            "initial_lr": 0.00013921427757316638,
            "max_epochs": 3,
            "lr_reduction_factor": 0.6718664781335942,
            "lr_reduction_patience": 3,
            "early_stop_patience": 6,
            "batch_size": 64
        }
        RANDOM_STATE = 123
        EXPECTED_ACCURACY=0.9
        df = self.prepare_dataset(DATAPATH)
        train_subsets, val_subsets = self.split_dataset(df, RANDOM_STATE)

        texts = df['sentence'].values

        tf.keras.backend.clear_session()
        clf = ContextBranchSentenceClassifier(
            model_params=model_params,
            corpus=texts,
            bod_line=BOM_SIGNAL,
            eod_line=EOM_SIGNAL
        )
        clf.prepare_train_records(data=train_subsets[0])
        clf.prepare_validation_records(data=val_subsets[0])
        clf.compile(optimizer=tf.keras.optimizers.Adam(model_params['initial_lr']),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy'])
        clf.fit(
            x=(clf.train_texts[:, 0], clf.train_texts[:, 1], clf.train_texts[:, 2]),
            y=clf.train_labels,
            batch_size=model_params['batch_size'],
            epochs=model_params['max_epochs'],
            validation_data=((clf.validation_texts[:, 0], clf.validation_texts[:, 1], clf.validation_texts[:, 2]),
                             clf.validation_labels),
            use_multiprocessing=True,
            callbacks=[
                tf.keras.callbacks.ReduceLROnPlateau(
                    factor=model_params['lr_reduction_factor'], patience=model_params['lr_reduction_patience'],
                    verbose=0),
                tf.keras.callbacks.EarlyStopping(
                    min_delta=1e-4, patience=model_params['early_stop_patience'], restore_best_weights=True)
            ], verbose=1)
        eval_loss, eval_accuracy = clf.evaluate(
            (clf.validation_texts[:, 0], clf.validation_texts[:, 1], clf.validation_texts[:, 2]),
            clf.validation_labels)

        print(f"Accuracy: {eval_accuracy:.4f}")
        self.assertGreaterEqual(eval_accuracy, EXPECTED_ACCURACY,
                                f'Obtained accuracy {eval_accuracy:.4f} less than expected {EXPECTED_ACCURACY:.4f}')

if __name__ == '__main__':
    unittest.main()
