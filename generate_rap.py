import tensorflow as tf
import numpy as np
import os
from preprocess_and_train import build_model
from preprocess_and_train import vocab_size, embedding_dim, rnn_units, checkpoint_dir, char2num, num2char


def generate_text(model, start_string):
    num_generate = 300
    input_data = [char2num[c] for c in start_string]
    input_data = tf.expand_dims(input_data, 0)

    model.reset_states()

    text_generated = []
    
    for i in range(num_generate):
        prediction = model(input_data)
        prediction = tf.squeeze(prediction, 0)
        temperature = 1.0
        predicted_id = tf.random.categorical(prediction, num_samples=1)[-1,0].numpy()
        input_data = tf.expand_dims([predicted_id], 0)
        text_generated.append(num2char[predicted_id])
    
    return start_string+''.join(text_generated)

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

rap = generate_text(model, start_string = " ")

print(rap)

file = open('generatedrap.txt', 'w')
file.write(rap)
file.close()