
# from sklearn.base import BaseEstimator
import logging
import os
import numpy as np
import tensorflow as tf
import time
from src.models.data_manager import DataManager

#
# Sequence to Sequence Model
# LSTM based model with Attention
#


class SeqToSeqModel:
    """Implementation of the Sequence to Sequence model."""

    def __init__(self,
                 epochs=100,
                 batch_size=128,
                 rnn_size=512,
                 num_layers=2,
                 encoding_embedding_size=512,
                 decoding_embedding_size=512,
                 learning_rate=0.005,
                 learning_rate_decay=0.9,
                 min_learning_rate=0.0001,
                 keep_probability=0.75,
                 max_sequence_length=21,
                 beam_width =5,
                 model_name=None
                 ):

        # Model Parameters
        # Number of epochs
        self.epochs = epochs

        # Batch size
        self.batch_size = batch_size

        # Number of units in LSTM cell
        self.rnn_size = rnn_size

        # Number of RNN Layers
        self.num_layers = num_layers

        # Encoder embedding size
        self.encoding_embedding_size = encoding_embedding_size

        # Decoder embedding size
        self.decoding_embedding_size = decoding_embedding_size

        # Learning rate
        self.learning_rate = learning_rate

        # Learning rate decay
        self.learning_rate_decay = learning_rate_decay

        # Minimum Learning rate
        self.min_learning_rate = min_learning_rate

        # Keep probability
        self.keep_probability = keep_probability

        # Maximum sequence length
        self.max_sequence_length = max_sequence_length

        # Beam width
        self.beam_width = beam_width

        # Model Name
        self.model_name = model_name

        logging.info('Model SeqtoSeq Initialization completed.')

    def fit(self, data_manager: DataManager):
        """Training model."""

        sorted_questions = data_manager.sorted_questions
        sorted_answers = data_manager.sorted_answers

        # Start the session
        with tf.Graph().as_default(), tf.Session() as session:

            # Load the model inputs
            input_data, targets, lr, input_sequence_length, output_sequence_length = self.model_inputs()

            # Model Graph Variables
            self.model_graph_vars(data_manager)

            # Create Encoder
            encoder_output, encoder_state = self._get_encoder(input_data, input_sequence_length)

            # Create Decoder for Training
            train_logits = self._get_decoder_train(
                targets, encoder_output, encoder_state, input_sequence_length, data_manager)

            with tf.name_scope("optimization"):
                # Compute weight mask
                mask = tf.sequence_mask(output_sequence_length, self.max_sequence_length, dtype=tf.float32)

                # Loss function
                cost = tf.contrib.seq2seq.sequence_loss(
                    train_logits,
                    targets,
                    mask)

                # Optimizer
                optimizer = tf.train.AdamOptimizer(self.learning_rate)

                # Gradient Clipping
                gradients = optimizer.compute_gradients(cost)
                capped_gradients = [
                    (tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
                train_op = optimizer.apply_gradients(capped_gradients)

            session.run(tf.global_variables_initializer())

            # Train
            summary_train_loss, summary_valid_loss = self.train(
                session, sorted_questions, sorted_answers, train_op, cost, data_manager)

            logging.info('Summary Train Loss: \n{0}'.format(summary_train_loss))
            logging.info('Summary Valid Loss: \n{0}'.format(summary_valid_loss))

        return

    def train(self, session, sorted_questions, sorted_answers, train_op, cost, data_manager: DataManager, save=True):

        questions_vocab_to_int = data_manager.questions_vocab_to_int
        answers_vocab_to_int = data_manager.answers_vocab_to_int

        # Validate the training with 10% of the data
        train_valid_split = int(len(sorted_questions) * 0.15)

        # Split the questions and answers into training and validating data
        train_questions = sorted_questions[train_valid_split:]
        train_answers = sorted_answers[train_valid_split:]

        valid_questions = sorted_questions[:train_valid_split]
        valid_answers = sorted_answers[:train_valid_split]

        # Check training loss after every 100 batches
        display_step = 100

        # Early Stop Initialization
        stop_early = 0

        # If the validation loss does decrease in 5 consecutive checks, stop training
        stop = 5

        # Modulus for checking validation loss
        validation_check = ((len(train_questions)) // self.batch_size // 2) - 1

        # Record the training loss for each display step
        total_train_loss = 0
        total_summary_train_loss = 0.
        min_valid_loss = 0.

        # Record the validation loss for saving improvements in the model
        summary_train_loss = []
        summary_valid_loss = []
        learning_rate = self.learning_rate

        # Get Placeholders
        graph = tf.get_default_graph()
        input_data = graph.get_tensor_by_name('Inputs/input_data:0')
        targets = graph.get_tensor_by_name('Inputs/targets:0')
        lr = graph.get_tensor_by_name('Inputs/learning_rate:0')
        input_sequence_length = graph.get_tensor_by_name('Inputs/input_sequence_length:0')
        output_sequence_length = graph.get_tensor_by_name('Inputs/output_sequence_length:0')

        for epoch_i in range(1, self.epochs + 1):
            shuffled_questions, shuffled_answers = self._shuffle_training_data(train_questions, train_answers)

            for batch_i, \
                (questions_batch, answers_batch, q_sequence_length_batch, a_sequence_length_batch) in enumerate(
                 self.batch_data(shuffled_questions,
                                 shuffled_answers,
                                 self.batch_size,
                                 questions_vocab_to_int,
                                 answers_vocab_to_int)):

                feed_dict = {
                    input_data: questions_batch,
                    targets: answers_batch,
                    lr: learning_rate,
                    input_sequence_length: q_sequence_length_batch,
                    output_sequence_length: a_sequence_length_batch
                }

                start_time = time.time()
                _, loss = session.run([train_op, cost], feed_dict=feed_dict)

                total_train_loss += loss
                total_summary_train_loss += loss
                end_time = time.time()
                batch_time = end_time - start_time

                if batch_i % display_step == 0:
                    print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                          .format(epoch_i,
                                  self.epochs,
                                  batch_i,
                                  len(train_questions) // self.batch_size,
                                  total_train_loss / display_step,
                                  batch_time * display_step))
                    logging.info('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                                 .format(epoch_i,
                                         self.epochs,
                                         batch_i,
                                         len(train_questions) // self.batch_size,
                                         total_train_loss / display_step,
                                         batch_time * display_step))
                    total_train_loss = 0

                if batch_i % validation_check == 0 and batch_i > 0:
                    total_valid_loss = 0
                    start_time = time.time()
                    for batch_ii, \
                        (questions_batch_ii, answers_batch_ii,
                         q_sequence_length_batch_ii, a_sequence_length_batch_ii) in \
                            enumerate(self.batch_data(valid_questions, valid_answers, self.batch_size,
                                                      questions_vocab_to_int, answers_vocab_to_int)):
                        valid_loss = session.run(
                            cost, {input_data: questions_batch_ii,
                                   targets: answers_batch_ii,
                                   lr: learning_rate,
                                   input_sequence_length: q_sequence_length_batch_ii,
                                   output_sequence_length: a_sequence_length_batch_ii})
                        total_valid_loss += valid_loss
                    end_time = time.time()
                    batch_time = end_time - start_time
                    avg_valid_loss = total_valid_loss / (len(valid_questions) / self.batch_size)
                    print('Valid Loss: {:>6.3f}, Seconds: {:>5.2f}'.format(avg_valid_loss, batch_time))
                    logging.info('Valid Loss: {:>6.3f}, Seconds: {:>5.2f}'.format(avg_valid_loss, batch_time))

                    avg_train_loss = total_summary_train_loss / (validation_check * self.batch_size)
                    summary_train_loss.append(avg_train_loss)

                    # Reduce learning rate, but not below its minimum value
                    learning_rate *= self.learning_rate_decay
                    if learning_rate < self.min_learning_rate:
                        learning_rate = self.min_learning_rate

                    summary_valid_loss.append(avg_valid_loss)
                    if len(summary_valid_loss) == 1 or min_valid_loss > avg_valid_loss:
                        print('New Record!')
                        logging.info('New Record!')
                        min_valid_loss = avg_valid_loss
                        stop_early = 0
                        if save:
                            self._save_model(session)

                    else:
                        print("No Improvement.")
                        logging.info("No Improvement.")
                        stop_early += 1
                        if stop_early == stop:
                            break

            if stop_early == stop:
                print("Stopping Training.")
                logging.info("Stopping Training.")
                break

        return summary_train_loss, summary_valid_loss

    def predict(self, input_question, data_manager: DataManager):
        """Predict a response for the given question"""

        # Reset the graph to ensure that it is ready for training
        tf.reset_default_graph()

        # Start the session
        with tf.Session() as session:

            # Load the model inputs
            input_data, targets, lr, input_sequence_length, output_sequence_length = self.model_inputs()

            # Model Graph Variables
            self.model_graph_vars(data_manager)

            # Create Encoder
            encoder_output, encoder_state = self._get_encoder(input_data, input_sequence_length)

            # Create Decoder for Training
            infer_logits = self._get_decoder_infer(
                encoder_output, encoder_state, input_sequence_length, data_manager)

            session.run(tf.global_variables_initializer())

            # Restore session
            saver = tf.train.Saver()
            save_path = self._get_model_save_path()
            saver.restore(session, save_path)
            logging.info('Saved model {0} loaded from disk.'.format(save_path))

            # Add empty questions so the the input_data is the correct shape
            questions = [input_question] * self.batch_size

            # Add empty answers so the the input_data is the correct shape
            single_input_answer = [data_manager.answers_vocab_to_int['<GO>'],
                                   data_manager.answers_vocab_to_int['<PAD>']]
            answers = [single_input_answer] * self.batch_size

            # Get Placeholders
            graph = tf.get_default_graph()
            input_data = graph.get_tensor_by_name('Inputs/input_data:0')
            targets = graph.get_tensor_by_name('Inputs/targets:0')
            lr = graph.get_tensor_by_name('Inputs/learning_rate:0')
            input_sequence_length = graph.get_tensor_by_name('Inputs/input_sequence_length:0')
            output_sequence_length = graph.get_tensor_by_name('Inputs/output_sequence_length:0')

            for batch_i, \
                (pad_questions_batch, pad_answers_batch, q_sequence_length_batch, a_sequence_length_batch) in enumerate(
                    self.batch_data(questions, answers, self.batch_size,
                                    data_manager.questions_vocab_to_int, data_manager.answers_vocab_to_int)):

                # Build feed_dict
                feed_dict = {
                    input_data: pad_questions_batch,
                    targets: pad_answers_batch,
                    lr: self.learning_rate,
                    input_sequence_length: q_sequence_length_batch,
                    output_sequence_length: a_sequence_length_batch
                }

                # Get prediction
                answers = session.run([infer_logits], feed_dict=feed_dict)
                answer = np.argmax(answers[0][0], axis=-1)

                return answer

        raise AssertionError('Prediction batch processing failed.')

    def predict_beam_load_model(self, session, data_manager: DataManager):
        """Loads model for Predict beam method."""

        # Load the model inputs
        input_data, targets, lr, input_sequence_length, output_sequence_length = self.model_inputs()

        # Model Graph Variables
        self.model_graph_vars(data_manager)

        # Create Encoder
        encoder_output, encoder_state = self._get_encoder(input_data, input_sequence_length)

        # Create Decoder for Training
        beam_output = self._get_decoder_infer_beam(
            encoder_output, encoder_state, input_sequence_length, data_manager)

        session.run(tf.global_variables_initializer())

        # Restore session
        saver = tf.train.Saver()
        save_path = self._get_model_save_path()
        saver.restore(session, save_path)
        logging.info('Saved model {0} loaded from disk.'.format(save_path))

        return beam_output

    def predict_beam_responses(self, session, beam_output, input_question, data_manager: DataManager):
        """Gets beam responses for given question."""
        # Add empty questions so the the input_data is the correct shape
        questions = [input_question] * self.batch_size

        # Add empty answers so the the input_data is the correct shape
        single_input_answer = [data_manager.answers_vocab_to_int['<GO>'],
                               data_manager.answers_vocab_to_int['<PAD>']]
        answers = [single_input_answer] * self.batch_size

        # Get Placeholders
        graph = tf.get_default_graph()
        input_data = graph.get_tensor_by_name('Inputs/input_data:0')
        targets = graph.get_tensor_by_name('Inputs/targets:0')
        lr = graph.get_tensor_by_name('Inputs/learning_rate:0')
        input_sequence_length = graph.get_tensor_by_name('Inputs/input_sequence_length:0')
        output_sequence_length = graph.get_tensor_by_name('Inputs/output_sequence_length:0')

        for batch_i, \
            (pad_questions_batch, pad_answers_batch, q_sequence_length_batch, a_sequence_length_batch) in enumerate(
             self.batch_data(questions, answers, self.batch_size,
                             data_manager.questions_vocab_to_int, data_manager.answers_vocab_to_int)):
            # Build feed_dict
            feed_dict = {
                input_data: pad_questions_batch,
                targets: pad_answers_batch,
                lr: self.learning_rate,
                input_sequence_length: q_sequence_length_batch,
                output_sequence_length: a_sequence_length_batch
            }

            # Get prediction
            output = session.run([beam_output], feed_dict=feed_dict)
            return output[0].scores, output[0].predicted_ids, output[0].parent_ids

        raise AssertionError('Prediction batch processing failed.')

    def predict_beam(self, input_question, data_manager: DataManager):
        """Predict a response for the given question"""

        # Reset the graph to ensure that it is ready for training
        tf.reset_default_graph()

        # Start the session
        with tf.Session() as session:

            # Load model
            beam_output = self.predict_beam_load_model(session, data_manager)

            # Predict beam responses
            return self.predict_beam_responses(session, beam_output, input_question, data_manager)

    def model_inputs(self):
        """Create placeholders for inputs to the model"""

        with tf.name_scope('Inputs'):

            # Question
            input_data = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_sequence_length], name='input_data')

            # Response
            targets = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_sequence_length], name='targets')

            # Learning Rate
            lr = tf.placeholder(tf.float32, name='learning_rate')

            # Input Sequence length
            input_sequence_length = tf.placeholder(tf.int32, shape=[self.batch_size], name="input_sequence_length")

            # Output Sequence length
            output_sequence_length = tf.placeholder(tf.int32, shape=[self.batch_size], name="output_sequence_length")

        return input_data, targets, lr, input_sequence_length, output_sequence_length

    def model_graph_vars(self, data_manager: DataManager):
        """Create Graph variables for this model"""

        uniform_r = 0.01

        # Build Word Embeddings.
        with tf.variable_scope('Embeddings', reuse=tf.AUTO_REUSE):
            _ = tf.get_variable(name='Input_Embeddings',
                                shape=[len(data_manager.questions_int_to_vocab), self.encoding_embedding_size],
                                initializer=tf.random_uniform_initializer(-1 * uniform_r, uniform_r),
                                trainable=True)

            _ = tf.get_variable(name='Output_Embeddings',
                                shape=[len(data_manager.answers_int_to_vocab), self.decoding_embedding_size],
                                initializer=tf.random_uniform_initializer(-1 * uniform_r, uniform_r),
                                trainable=True)

    @staticmethod
    def process_encoding_input(target_data, go_token, batch_size):
        """Remove the last word id from each batch and concat the <GO> to the begining of each batch"""
        ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([batch_size, 1], go_token), ending], 1)

        return dec_input

    def decode(self, vocab_size, input_sequence_length, encoder_output, encoder_state, helper, scope, reuse=None):

        with tf.variable_scope(scope, reuse=reuse):
            lstm = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
            drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=self.keep_probability)
            dec_cell = tf.contrib.rnn.MultiRNNCell([drop] * self.num_layers)

            # alternatively concat forward and backward states
            bi_encoder_state = []
            for layer_id in range(self.num_layers):
                bi_encoder_state.append(encoder_state[0][layer_id])  # forward
                # bi_encoder_state.append(encoder_state[1][layer_id])  # backward

            bi_encoder_state = tuple(bi_encoder_state)

            # Create Attention Mechanism
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=self.rnn_size,
                memory=tf.concat(encoder_output, -1),
                memory_sequence_length=input_sequence_length)

            # # Function to combine inputs and attention
            # def add_attention(inputs, attention):
            #     f, b = tf.split(value=attention, num_or_size_splits=2, axis=tf.constant(1, dtype=tf.int32))
            #     return tf.multiply(inputs, tf.add(f, b))

            attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                dec_cell,
                attention_mechanism,
                cell_input_fn=lambda inputs, attention: inputs,
                output_attention=False)

            attention_zero = attn_cell.zero_state(
                batch_size=self.batch_size,
                dtype=tf.float32).clone(cell_state=bi_encoder_state)

            projection_layer = tf.layers.Dense(vocab_size, use_bias=True, bias_initializer=tf.zeros_initializer())

            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=attn_cell,
                helper=helper,
                initial_state=attention_zero,
                output_layer=projection_layer
            )

            final_outputs, _final_state, _final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder,
                impute_finished=True,
                maximum_iterations=self.max_sequence_length
            )

            logits = final_outputs.rnn_output

            return logits

    def _get_encoder(self, input_data, input_sequence_length):
        """Creates an encoder instance"""

        graph = tf.get_default_graph()
        enc_embeddings = graph.get_tensor_by_name('Embeddings/Input_Embeddings:0')
        enc_embed_input = tf.nn.embedding_lookup(enc_embeddings, input_data)

        lstm = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=self.keep_probability)
        enc_cell = tf.contrib.rnn.MultiRNNCell([drop] * self.num_layers)

        enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=enc_cell,
            cell_bw=enc_cell,
            sequence_length=input_sequence_length,
            inputs=enc_embed_input,
            dtype=tf.float32)

        return enc_output, enc_state

    def _get_decoder_train(self, target_data, encoder_output, encoder_state, input_sequence_length,
                           data_manager: DataManager):
        """Builds a decoder for training"""

        vocab_to_int = data_manager.questions_vocab_to_int
        start_of_sequence_id = vocab_to_int['<GO>']
        answers_vocab_size = len(data_manager.answers_vocab_to_int)

        dec_input = self.process_encoding_input(target_data, start_of_sequence_id, self.batch_size)
        graph = tf.get_default_graph()
        dec_embeddings = graph.get_tensor_by_name('Embeddings/Output_Embeddings:0')
        dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

        sequence_lengths = tf.fill([self.batch_size], self.max_sequence_length)

        train_helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, sequence_lengths)

        with tf.variable_scope("decoding"):

            train_logits = self.decode(
                answers_vocab_size, input_sequence_length, encoder_output, encoder_state, train_helper, 'decoding')

        return train_logits

    def _get_decoder_infer(self, encoder_output, encoder_state, input_sequence_length, data_manager: DataManager):
        """Gets a decoder for inferring a single result."""

        vocab_to_int = data_manager.questions_vocab_to_int
        start_of_sequence_id = vocab_to_int['<GO>']
        answers_vocab_size = len(data_manager.answers_vocab_to_int)

        start_tokens = tf.fill([self.batch_size], start_of_sequence_id)
        end_of_sequence_id = vocab_to_int['<EOS>']

        graph = tf.get_default_graph()
        dec_embeddings = graph.get_tensor_by_name('Embeddings/Output_Embeddings:0')

        infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings,
                                                                start_tokens=tf.to_int32(start_tokens),
                                                                end_token=end_of_sequence_id)

        with tf.variable_scope("decoding", reuse=tf.AUTO_REUSE):

            infer_logits = self.decode(
                answers_vocab_size, input_sequence_length, encoder_output, encoder_state, infer_helper, 'decoding',
                reuse=tf.AUTO_REUSE)

        return infer_logits

    def decode_beam(self, vocab_size, input_sequence_length, encoder_output, encoder_state,
                    dec_embeddings, start_token, end_token, scope, reuse=None):

        beam_width = self.beam_width

        with tf.variable_scope(scope, reuse=reuse):
            lstm = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
            drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=self.keep_probability)
            dec_cell = tf.contrib.rnn.MultiRNNCell([drop] * self.num_layers)

            # alternatively concat forward and backward states
            bi_encoder_state = []
            for layer_id in range(self.num_layers):
                bi_encoder_state.append(encoder_state[0][layer_id])  # forward
                # bi_encoder_state.append(encoder_state[1][layer_id])  # backward

            bi_encoder_state = tuple(bi_encoder_state)

            tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(tf.concat(encoder_output, -1), multiplier=beam_width)
            tiled_encoder_state = tf.contrib.seq2seq.tile_batch(bi_encoder_state, multiplier=beam_width)
            tiled_sequence_length = tf.contrib.seq2seq.tile_batch(input_sequence_length, multiplier=beam_width)

            # Create Attention Mechanism
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=self.rnn_size,
                memory=tiled_encoder_outputs,
                memory_sequence_length=tiled_sequence_length)

            # # Function to combine inputs and attention
            # def add_attention(inputs, attention):
            #     f, b = tf.split(value=attention, num_or_size_splits=2, axis=tf.constant(1, dtype=tf.int32))
            #     return tf.multiply(inputs, tf.add(f, b))

            attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                dec_cell,
                attention_mechanism,
                cell_input_fn=lambda inputs, attention: inputs,
                output_attention=False)

            attention_zero = attn_cell.zero_state(
                batch_size=self.batch_size * beam_width,
                dtype=tf.float32).clone(cell_state=tiled_encoder_state)

            projection_layer = tf.layers.Dense(vocab_size, use_bias=True, bias_initializer=tf.zeros_initializer())

            # Define a beam-search decoder
            beam_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=attn_cell,
                embedding=dec_embeddings,
                start_tokens=tf.fill([self.batch_size], start_token),
                end_token=end_token,
                initial_state=attention_zero,
                beam_width=self.beam_width,
                output_layer=projection_layer,
                length_penalty_weight=0.0)

            final_outputs, _final_state, _final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder=beam_decoder,
                # impute_finished=True,
                maximum_iterations=self.max_sequence_length
            )

            return final_outputs.beam_search_decoder_output

    def _get_decoder_infer_beam(self, encoder_output, encoder_state, input_sequence_length, data_manager: DataManager):
        """Gets Decoder for Beam Search."""

        vocab_to_int = data_manager.questions_vocab_to_int
        start_of_sequence_id = vocab_to_int['<GO>']
        answers_vocab_size = len(data_manager.answers_vocab_to_int)
        end_of_sequence_id = vocab_to_int['<EOS>']

        graph = tf.get_default_graph()
        dec_embeddings = graph.get_tensor_by_name('Embeddings/Output_Embeddings:0')

        with tf.variable_scope("decoding", reuse=tf.AUTO_REUSE):

            return self.decode_beam(
                answers_vocab_size, input_sequence_length, encoder_output, encoder_state,
                dec_embeddings, start_of_sequence_id, end_of_sequence_id, 'decoding', reuse=tf.AUTO_REUSE)

    def get_encoded_representation(self, session, encoder_output, input_question, data_manager: DataManager):
        """Gets beam responses for given question."""

        if input_question is None:
            return None

        # Add empty questions so the the input_data is the correct shape
        questions = [input_question] * self.batch_size

        # Add empty answers so the the input_data is the correct shape
        single_input_answer = [data_manager.answers_vocab_to_int['<GO>'],
                               data_manager.answers_vocab_to_int['<PAD>']]
        answers = [single_input_answer] * self.batch_size

        # Get Placeholders
        graph = tf.get_default_graph()
        input_data = graph.get_tensor_by_name('Inputs/input_data:0')
        targets = graph.get_tensor_by_name('Inputs/targets:0')
        lr = graph.get_tensor_by_name('Inputs/learning_rate:0')
        input_sequence_length = graph.get_tensor_by_name('Inputs/input_sequence_length:0')
        output_sequence_length = graph.get_tensor_by_name('Inputs/output_sequence_length:0')

        for batch_i, \
            (pad_questions_batch, pad_answers_batch, q_sequence_length_batch, a_sequence_length_batch) in enumerate(
             self.batch_data(questions, answers, self.batch_size,
                             data_manager.questions_vocab_to_int, data_manager.answers_vocab_to_int)):
            # Build feed_dict
            feed_dict = {
                input_data: pad_questions_batch,
                targets: pad_answers_batch,
                lr: self.learning_rate,
                input_sequence_length: q_sequence_length_batch,
                output_sequence_length: a_sequence_length_batch
            }

            # Get prediction
            output = session.run([encoder_output], feed_dict=feed_dict)
            return output[0]

        raise AssertionError('Get Encoder Representation failed.')

    def pad_sentence_batch(self, sentence_batch, pad_token):
        """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
        max_sentence = self.max_sequence_length
        return [sentence + [pad_token] * (max_sentence - len(sentence)) for sentence in sentence_batch]

    def batch_data(self, questions, answers, batch_size, questions_vocab_to_int, answers_vocab_to_int):
        """Batch questions and answers together"""
        for batch_i in range(0, len(questions) // batch_size):
            start_i = batch_i * batch_size
            questions_batch = questions[start_i:start_i + batch_size]
            answers_batch = answers[start_i:start_i + batch_size]
            q_sequence_length_batch = [len(question) for question in questions_batch]
            a_sequence_length_batch = [len(answer) for answer in answers_batch]
            pad_questions_batch = self.pad_sentence_batch(questions_batch, questions_vocab_to_int['<PAD>'])
            pad_answers_batch = self.pad_sentence_batch(answers_batch, answers_vocab_to_int['<PAD>'])
            yield pad_questions_batch, pad_answers_batch, q_sequence_length_batch, a_sequence_length_batch

    def _save_model(self, session):
        """ Saves model to the disk. Should be called only by fit.

        :param session:
            Valid session object.
        :return:
            None.
        """

        # Save model for tensorflow reuse for next epoch
        saver = tf.train.Saver()
        save_path = self._get_model_save_path()
        saver.save(session, save_path)
        logging.info('Saved model {0} to disk.'.format(save_path))

    def _get_save_dir(self):
        """ Checks for save directory and builds it if necessary.

        :return:
             A string containing save directory path
        """
        abs_path = os.path.abspath(os.path.dirname(__file__))
        def_models_path = os.path.join(abs_path, '../../models/')
        save_dir = def_models_path + self.model_name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        return save_dir

    def _get_model_save_path(self):
        """ Builds save path for the model.

        :return:
            A string containing model save path.
        """
        assert self.model_name is not None
        return '{0}/{1}.ckpt'.format(self._get_save_dir(), self.model_name)

    @staticmethod
    def _shuffle_training_data(questions, answers):
        """Randomly shuffles questions and answers training data."""

        index = np.arange(len(questions))
        np.random.shuffle(index)

        q = []
        a = []
        for i in index:
            q.append(questions[i])
            a.append(answers[i])

        return q, a
