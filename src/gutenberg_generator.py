# import stuff
import tqdm
import tensorflow as tf
import json
import numpy as np
import pandas as pd
import os
import time
import argparse
import keras

def load_and_process(train_size):
    # loading in the gutenberg-poetry corpus
    filepath = os.path.join("in", "gutenberg-poetry-v001.ndjson")
    text = open(filepath, 'rb').read().decode(encoding='utf-8')
    text_df = pd.read_json(filepath, lines=True)
    # defining the training data
    text_train = text_df[:int(train_size)]
    # merging the text
    merged_text = text_train["s"].sum()
    # getting the unique characters in the training data
    vocab = sorted(set(merged_text))
    # split the text up into tokens
    chars = tf.strings.unicode_split(merged_text, input_encoding='UTF-8')
    # using a StringLookup layer to convert characters to numerical IDs
    ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
    # this converts from tokens to character ids
    ids = ids_from_chars(chars)
    # this inverts that process
    chars_from_ids = tf.keras.layers.StringLookup(vocabulary = ids_from_chars.get_vocabulary(), invert = True, mask_token = None)
    chars = chars_from_ids(ids)
    return merged_text, vocab, chars, ids_from_chars, ids, chars_from_ids

# defining a function that recovers text from numerical ids
def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis = -1)

# define a function that splits sequences up into input and label
def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

# create training data
def data_preparation(ids_from_chars, merged_text, b_size):
    # get all ids
    all_ids = ids_from_chars(tf.strings.unicode_split(merged_text, "UTF-8"))
    # make a dataset of ids
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
    # define the length of sequences
    seq_length = 100
    # batching the sequences to the desired size
    sequences = ids_dataset.batch(seq_length+1, drop_remainder = True)
    # split the sequences up into input and label
    dataset = sequences.map(split_input_target) # this is a different way to iterate a function over your data than what we've done in class
    # create training batches
    BUFFER_SIZE = 10000
    # batch the dataset
    dataset = (dataset
               .shuffle(BUFFER_SIZE)
               .batch(int(b_size), drop_remainder = True)
               .prefetch(tf.data.experimental.AUTOTUNE)) # Whichever dataset pre-fetches are set to AUTOTUNE are monitored and optimized
    return dataset

# build your model
# This is object oriented programming, which we haven't seen too much of in class
def build_model(ids_from_chars, embed_dim):
    # get StringLookup vocab size
    vocab_size = len(ids_from_chars.get_vocabulary())
    # define the embedding dimension
    embedding_dim = int(embed_dim)
    # number of RNN units
    rnn_units = 1024
    # build the MyModel object - this is lifted from https://www.tensorflow.org/text/tutorials/text_generation - on which this project is based
    class MyModel(tf.keras.Model):
        def __init__(self, vocab_size, embedding_dim, rnn_units): #initializing the object
            super().__init__(self)
            # defining the embedding layer
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim) 
            # defining the RNN layer
            self.gru = tf.keras.layers.GRU(rnn_units,
                                                 return_sequences = True,
                                                 return_state = True)
            # defining the output layer
            self.dense = tf.keras.layers.Dense(vocab_size)
        
        # defining a function that maintains the internal state of the model
        def call(self, inputs, states = None, return_state = False, training = False):
            x = inputs
            x = self.embedding(x, training = training)
            if states is None:
                states = self.gru.get_initial_state(x)
            x, states = self.gru(x, initial_state = states, training = training)
            x = self.dense(x, training = training)
            
            if return_state:
                return x, states
            else:
                return x
            

    
    model = MyModel(vocab_size = vocab_size,
                    embedding_dim = embedding_dim,
                    rnn_units = rnn_units)
    return model

# train the model
def compile_train_mdl(model, learn_rate, dataset, eps):
    # define a loss function
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True) # the from_logits flag since our predictions are returned in logits
    model.compile(optimizer = keras.optimizers.Adam(learning_rate= float(learn_rate)),
                  loss = loss)
    history = model.fit(dataset, epochs = int(eps))
    return model, history

# make a single step predictor, again using something lifted from https://www.tensorflow.org/text/tutorials/text_generation

def predictor(model, chars_from_ids, ids_from_chars):
    class OneStep(tf.keras.Model):
        def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
            super().__init__()
            self.temperature = temperature
            self.model = model
            self.chars_from_ids = chars_from_ids
            self.ids_from_chars = ids_from_chars

            # Create a mask to prevent "[UNK]" from being generated.
            skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
            sparse_mask = tf.SparseTensor(
                # Put a -inf at each bad index.
                values=[-float('inf')]*len(skip_ids),
                indices=skip_ids,
                # Match the shape to the vocabulary
                dense_shape=[len(ids_from_chars.get_vocabulary())])
            self.prediction_mask = tf.sparse.to_dense(sparse_mask)
        
        @tf.function
        def generate_one_step(self, inputs, states=None):
            # Convert strings to token IDs.
            input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
            input_ids = self.ids_from_chars(input_chars).to_tensor()

            # Run the model.
            # predicted_logits.shape is [batch, char, next_char_logits]
            predicted_logits, states = self.model(inputs=input_ids, states=states,
                                          return_state=True)
            # Only use the last prediction.
            predicted_logits = predicted_logits[:, -1, :]
            predicted_logits = predicted_logits/self.temperature
            # Apply the prediction mask: prevent "[UNK]" from being generated.
            predicted_logits = predicted_logits + self.prediction_mask

            # Sample the output logits to generate token IDs.
            predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
            predicted_ids = tf.squeeze(predicted_ids, axis=-1)

            # Convert from token ids to characters
            predicted_chars = self.chars_from_ids(predicted_ids)

            # Return the characters and model state.
            return predicted_chars, states
        
    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
    return one_step_model

def generate_text(start_word, gen_length, one_step_model, text_name):
    start = time.time()
    states = None
    # defining the starting point
    next_char = tf.constant([start_word])
    # the list to which each predicted word will be appended
    result = [next_char]
    
    # basically defining how much text you want to generate, counted in characters
    for n in range(int(gen_length)):
        next_char, states = one_step_model.generate_one_step(next_char, states = states)
        result.append(next_char)
        
    result = tf.strings.join(result)
    end = time.time()
    print(result[0].numpy().decode("utf-8"), "\n\n" + "_"*80)
    print("\Run time:", end - start)
    
    outpath = os.path.join("out", text_name)
    with open(outpath, "w") as text_file:
        text_file.write(str(result))
        
    return result

def parse_args():
    # initialize argparse
    ap = argparse.ArgumentParser()
    # add command line parameters
    ap.add_argument("-t", "--text_name", required=True, help="the name of the generated txt file")
    ap.add_argument("-em", "--embed_dim", required = True, help = "the size of the embedding layer of the model")
    ap.add_argument("-tr", "--train_size", required = True, help = "the size of the training dataset, measured in characters")
    ap.add_argument("-ep", "--eps", required=True, help="the amount of epochs you want the model to train")
    ap.add_argument("-bs", "--b_size", required=True, help="the batch size of the model")
    ap.add_argument("-lr", "--learn_rate", required=True, help="the learning rate of the model") # tensorflow recommends 0.001
    ap.add_argument("-gl", "--gen_length", required = True, help = "the length of the generated text file, measured in characters")
    ap.add_argument("-sw", "--start_word", required = True, help = "the first word of the generated text, which the model will make predictions on")
    args = vars(ap.parse_args())
    return args


def main():
    args = parse_args()
    merged_text, vocab, chars, ids_from_chars, ids, chars_from_ids = load_and_process(args["train_size"])
    dataset = data_preparation(ids_from_chars, merged_text, args["b_size"])
    model = build_model(ids_from_chars, args["embed_dim"])
    model, history = compile_train_mdl(model, args["learn_rate"], dataset, args["eps"])
    one_step_model = predictor(model, chars_from_ids, ids_from_chars)
    result = generate_text(args["start_word"], args["gen_length"], one_step_model, args["text_name"])
    
if __name__ == "__main__":
    main()
    


    
            
            
    
    
    
    
    
    
