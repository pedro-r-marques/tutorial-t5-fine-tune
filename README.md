# T5 fine-tuning on TPUs

This repository contains an example of how to fine tune a T5 model on TPUs
using [colab](https://colab.research.google.com) free tier. I put it together
since I found the need to aggregate information from several different
sources.

The approach is based on the Hugging Face
[TFT5Model](https://huggingface.co/transformers/model_doc/t5.html#tft5forconditionalgeneration) rather than the google research [repository](https://github.com/google-research/text-to-text-transfer-transformer).

The Hugging Face models can be used as standard Keras models and have support to load pre-trained weights.
However the existing tutorials that I found for the HF models use pytorch XLA and the HF trainer code.
Tensorflow/Keras has a much more complete and mature support to distribute models and training ops to
multiple TPUs.

This is a very fast moving echo-system and this tutorial will probably be outdated very soon.
As of October 2021 it seemed a reasonable way to fine-tune a T5 model on a text classification
problem.

## Dataset

For this tutorial, I used a [dataset](https://www.kaggle.com/imoore/60k-stack-overflow-questions-with-quality-rate) of 60k questions from stack-overflow rated as:

- Closed due to low quality
- Low quality (open)
- High quality

To download this dataset you can use the following command:

```sh
kaggle datasets download -d "imoore/60k-stack-overflow-questions-with-quality-rate"
unzip -d datasets 60k-stack-overflow-questions-with-quality-rate.zip
```

## Text classification task

The problem was modeled as predicting a single output token with a name of
class: "none", "low" or "high". Using a single token simplifies the prediction
process, since rather than a sentence which requires incremental prediction
passes, the code can predict simply the token outputed by the model when the
decoder inputs are `<bos>` (Beginging of sentence).

The input sequence was prefixed with `"quality:"` in order to differentiate
from other possible tasks. For instance, with the same dataset it is possible
to implement question tag prediction also. An interesting experiement would be
to mix both tasks in fine-tuning.

Questions have both `Title` and `Body` that is concatenated and fed has input
tokens to the model.

The T5 Model supports a maximum of 512 tokens. The average text of a question
exceeeds that limit. The code uses the first and last 256 tokens of the text.

## Dataset tokenization and pre-processing

One of the challenges of training large models, specially in the `colab`
infrastructure, is to load the examples to the TPU units without exhausting the
memory of the central unit.

TPUs support loading datasets in `TFRecord` format directly to the TPU and
running dataset processing operations directly in the TPU control units.

This approach minimizes the memory utilization in the central CPU. However it
requires loading the data from a Google Cloud Storage bucket. This tutorial
assumes a google cloud project and a google Storage Bucket that can be used
to move the data to the TPU control units.

A python script is provided, which can be run offline, to prepare the data and
generate files in `.tfrecord`. To generate the data files, install a python
(virtual) environment in your local workstation, initialize it with
`pip install -r requirements.txt`
and execute the following commands:

```sh
python dataset_tf_record_t5.py
gsutil cp datasets/dataset_t5_{train,valid}.tfrecord ${GCS_BUCKET}/so-quality/
```

This assumes that the workstation has access to the google cloud command line utils.

## Training (fine-tune)

The fine-tuning process is achieved by the script
[so_quality_train.ipynb](https://colab.research.google.com/drive/1zvsMcpK3KlckClNHI9SE5qcwqrnyTBmI?usp=sharing)
. This uses the generated `.tfrecord` files as `tf.data.Dataset`, loads a
pre-trained model (`t5-base`) and uses the `tf.keras.Model.fit` api to train
the model.

Tensorflow supports distributed training automatically under the
covers which spreads the load to the multiple TPU units available in `colab`.

The `Model.fit` API also supports checkpointing the weights of the model based
on a metric. It is useful here to use the classification accuracy on the validation
set to select the checkpoints to keep.

## Evaluation

With a trained model, the notebook
[so_quality_eval.ipynb](https://colab.research.google.com/drive/1ETqp8VPNQwhCaUnvlxmAC9mLl2-bw7T8?usp=sharing)
can be used to evaluate against the validation set and debug some examples.

In order for distributed prediction to work, it was necessary to wrap the
Hugging Face T5Model such that the `argmax` operation on logits is performed
on the TPU and only the token indices of the predicted classes are transfered
back to the central CPU. The default return object of the Hugging Face model
results in attempting to return all the `logits` back to the CPU.