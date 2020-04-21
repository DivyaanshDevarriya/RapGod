import tensorflow as tf
import numpy as np
import os

path_to_file = 'songs.txt'
seq_length = 30
BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
rnn_units = 1024 
char_to_filter = ['[', ']', '(', ')']
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

text = open(path_to_file, 'r').read()

for char in char_to_filter:
    text = text.replace(char, "")

vocab = sorted(set(text))

vocab_size = len(vocab)

char2num = {char:num for num,char in enumerate(vocab)}

num2char = np.array(vocab)

def split(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def build_model(vocab_size,embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape = [batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
        tf.keras.layers.GRU(rnn_units//2,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)

    ])
    return model

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

if __name__ == '__main__':
    
    text_as_num = np.array([char2num[char] for char in text])

    #print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_num[:13]))

    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_num)
    dataset = char_dataset.batch(seq_length+1, drop_remainder=True).map(split)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    model = build_model(vocab_size = vocab_size, embedding_dim = embedding_dim, rnn_units = rnn_units, batch_size = BATCH_SIZE)

    model.compile(optimizer = 'adam', loss = loss)

    print(model.summary())

    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_prefix,
        save_weights_only=True
    )

    patience = 3
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience)

    model.fit(dataset, epochs = 15, callbacks = [callback, early_stop])

