import datasets
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import tensorflow as tf

MAX_VOCAB_SIZE = 10000

def tf_lower_and_split_punct(text):
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, "[^ a-z.?!,¿]", "")
    text = tf.strings.regex_replace(text, "[.?!,¿]", r" \0 ")
    text = tf.strings.strip(text)
    text = tf.strings.join(["[SOS]", text, "[EOS]"], separator=" ")
    return text

def load_vectorizers(dataset, input_lang, output_lang):
    input_sentences = dataset[input_lang]
    output_sentences = dataset[output_lang]

    input_vectorizer = TextVectorization(
        max_tokens = MAX_VOCAB_SIZE,
        output_mode='int',
        standardize = tf_lower_and_split_punct
    )
    input_vectorizer.adapt(input_sentences)

    output_vectorizer = TextVectorization(
        max_tokens = MAX_VOCAB_SIZE,
        output_mode = 'int',
        standardize = tf_lower_and_split_punct
    )

    output_vectorizer.adapt(output_sentences)

    vectorizers = {input_lang: input_vectorizer, output_lang: output_vectorizer}
    return vectorizers

def load_data():

    dataset = datasets.load_dataset("ura-hcmut/PhoMT")
    merge_dataset = datasets.concatenate_datasets([dataset['train'],
                                                   dataset['test'],
                                                   dataset['validation']])

    def clean_examples(example):
        example['vi'] = example['vi'] if example['vi'] is not None else ''
        example['en'] = example['en'] if example['en'] is not None else ''
        return example

    dict_dataset = {}
    merge_dataset = merge_dataset.map(clean_examples)
    dict_dataset['en'] = merge_dataset['en'][:20000]
    dict_dataset['vi'] = merge_dataset['vi'][:20000]
    return dict_dataset

def preprocess_data(vectorizers, dataset, input_lang, output_lang):
    processed_dataset = {}

    vectorizer1 = vectorizers[input_lang]
    vectorizer2 = vectorizers[output_lang]

    text1 = dataset[input_lang]
    text2 = dataset[output_lang]

    processed_dataset[input_lang] = vectorizer1(text1)
    processed_dataset[output_lang] = vectorizer2(text2)

    return processed_dataset

def train_test_val_split(x, y, train_size, test_size):
    length = len(x)
    threshold1 = int(length * train_size)
    threshold2 = int(threshold1 + length * test_size)

    x_train, y_train = x[ : threshold1], y[ : threshold1]
    x_test, y_test = x[threshold1 : threshold2], y[threshold1 : threshold2]
    x_val, y_val = x[threshold2 : ], y[threshold2 : ]

    return [x_train, y_train, x_test, y_test, x_val, y_val]


def masked_loss(y_true, y_pred):

    loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)

    mask = tf.cast(y_true != 0, dtype = loss.dtype)

    loss *= mask
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)

def masked_acc(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)
    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)
    match*= mask

    return tf.reduce_sum(match)/tf.reduce_sum(mask)

def tokens_to_text(tokens, id_to_word):
    words = id_to_word(tokens)
    result = tf.strings.reduce_join(words, axis=-1, separator=" ")
    return result











