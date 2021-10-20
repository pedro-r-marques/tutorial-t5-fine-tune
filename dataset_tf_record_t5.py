""" Tokenize and encode the data in the TFRecord format.

TFRecord encoded data can be read directly by TPU processing nodes.
"""
import math
import random

from typing import List, Tuple

import tensorflow as tf
import pandas as pd

from tqdm import tqdm

from transformers import T5Tokenizer

CLASS_TOKENS = ['none', 'low', 'high']


def dataset_mapper(tokenizer, sep_token, example):
    """ Called for each example in order to implement manual truncation.
    """
    input_text = 'quality: ' + \
        example['Title'] + '</s>' + example['Body'] + '</s>'
    target_text = CLASS_TOKENS[example['class']]

    input_encodings = tokenizer.encode_plus(input_text)
    target_encodings = tokenizer.encode_plus(target_text, max_length=2)

    vec = input_encodings['input_ids']

    if len(vec) > 512:
        input_encodings['input_ids'] = vec[:255] + [sep_token] + vec[-256:]
        input_encodings['attention_mask'] = input_encodings['attention_mask'][:512]
    elif len(vec) < 512:
        pad = [0] * (512 - len(vec))
        input_encodings['input_ids'] = vec + pad
        input_encodings['attention_mask'] = input_encodings['attention_mask'] + pad

    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids'],
        'decoder_attention_mask': target_encodings['attention_mask']
    }

    return encodings


def _int64_list_feature(value: List[int]):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def make_tfrecord_dataset(tokenizer, df, filename):
    sep_token = tokenizer.get_vocab()['<extra_id_1>']

    with tf.io.TFRecordWriter(filename) as wr:
        for _, row in tqdm(df.iterrows()):
            encodings = dataset_mapper(tokenizer, sep_token, row)
            features = {k: _int64_list_feature(v)
                        for k, v in encodings.items()}
            features['class'] = _int64_list_feature([row['class']])
            example = tf.train.Example(
                features=tf.train.Features(feature=features))
            wr.write(example.SerializeToString())


def main():
    df_train = pd.read_csv('datasets/train.csv')
    df_valid = pd.read_csv('datasets/valid.csv')
    print('Train: %d records, Validation: %d records' %
          (df_train.shape[0], df_valid.shape[0]))

    class_map = {k: i for i, k in enumerate(['LQ_CLOSE', 'LQ_EDIT', 'HQ'])}
    df_train['class'] = df_train.Y.map(class_map)
    df_valid['class'] = df_valid.Y.map(class_map)

    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    make_tfrecord_dataset(tokenizer, df_train,
                          'datasets/dataset_t5_train.tfrecord')
    make_tfrecord_dataset(tokenizer, df_valid,
                          'datasets/dataset_t5_valid.tfrecord')


if __name__ == '__main__':
    main()
