import tensorflow as tf
from utils import *

class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, units):
        super(Encoder, self).__init__()
        #embedding layer
        self.embedding = tf.keras.layers.Embedding(input_dim = vocab_size, output_dim = units, mask_zero = True)


        #Rnn layer
        self.rnn = tf.keras.layers.Bidirectional(
            layer = tf.keras.layers.LSTM(units = units,
                                         return_sequences = True)
        )

    def call(self, input_tokens):

        x = self.embedding(input_tokens)

        x = self.rnn(x)

        return x


class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, dropout_rate = 0.3):

        super().__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads = num_heads,
            key_dim = key_dim,
            dropout = dropout_rate
        )

        self.add = tf.keras.layers.Add()
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, target, enc_output):

        attn_output = self.mha(
            query = target,
            value = enc_output
        )

        x = self.add([target, attn_output])

        x = self.norm(x)

        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, units, num_heads, dropout):
        super(Decoder, self).__init__()

        #this layer will output the embedding of the shifted-right target
        self.embedding = tf.keras.layers.Embedding(
            input_dim = vocab_size,
            output_dim = units,
            mask_zero = True
        )

        #This layer will output the context of the Embedded target
        self.pre_attention_rnn = tf.keras.layers.LSTM(
            units = units,
            return_sequences = True
        )

        #This attention layer computes context vectors by attending to the encoder output (context),
        #based on the decoder's current hidden states.
        self.attention = CrossAttention(num_heads = num_heads, key_dim = units, dropout_rate = dropout)

        #This layer processes the attended vectors to better model the combined context for generating output tokens
        self.post_attention_rnn = tf.keras.layers.LSTM(
            units = units,
            return_sequences = True
        )

        #this layer convert the information from the previous layer into the word probs that the model will choose from to generate
        self.output_layer = tf.keras.layers.Dense(units = vocab_size, activation = "log_softmax")

    def call(self, context, target):

        #embedding the target
        embedding_target = self.embedding(target)

        #learn the context of the target
        x = self.pre_attention_rnn(embedding_target)

        #learn the information that the model should pay attention to
        x = self.attention(target = x, enc_output = context)

        #learn the deeper context of the information
        x = self.post_attention_rnn(x)

        #convert to logits
        logits = self.output_layer(x)

        return logits

class Translator(tf.keras.Model):
    def __init__(self, input_vocab_size, output_vocab_size, units, num_heads, dropout = 0.3):
        super().__init__()

        self.encoder = Encoder(
            vocab_size = input_vocab_size,
            units = units
        )

        self.decoder = Decoder(
            vocab_size = output_vocab_size,
            units = units,
            num_heads=num_heads,
            dropout=dropout
        )

    def call(self, inputs):
        context, target = inputs
        encoded_context = self.encoder(context)

        logits = self.decoder(context = encoded_context, target = target)

        return logits

def compile_and_train(model, train_data, val_data, epochs=20, steps_per_epoch = 200):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.002), loss=masked_loss, metrics=[masked_acc, masked_loss])

    history = model.fit(
        train_data.repeat(),
        epochs=epochs,
        steps_per_epoch = steps_per_epoch,
        validation_data=val_data,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)],
    )

    return model, history




