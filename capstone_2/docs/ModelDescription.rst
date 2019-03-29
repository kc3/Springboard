
Model Description
~~~~~~~~~~~~~~~~~

Recurrent Neural Networks
^^^^^^^^^^^^^^^^^^^^^^^^^

Recurrent Neural Networks (RNN) are a particular kind of artificial
neural networks that are specialized in extracting information from
sequences. Unlike simple feedforward neural networks, a neuron’s
activation is also dependent from its previous activations. It allows
the model to capture correlations between the different inputs in the
sequence.In this project, we implement a sequence to sequence (Seq2Seq)
model using two independent RNNs, an encoder, and a decoder.

However, this type of architecture can be very challenging to train,
partly because it is much more prone to exploding and vanishing
gradients. These problems can be overcome by choosing more stable
activation functions like ReLU or tanh and by using more sophisticated
cells like LSTM and GRU, which involve more parameters and computations
than the vanilla RNN cell but are designed to avoid vanishing gradients
and capture long range dependencies. In this project, we make use of the
LSTM cell and gradient clipping to solve these problems.

In a regular RNN, at time-step :math:`t`, the cell state :math:`h_t` is
computed based on its own input and the cell state :math:`h_{t-1}` that
encapsulates some information from the precedent inputs: > :math:`h_t` =
:math:`f(W^{hx}x_t + W^{hh}h_{t-1})`

In this case :math:`f` is the activation function. The cell output for
this time-step is then computed using a third series of parameters
:math:`W^{hy}`: > :math:`y_t` = :math:`W^{hy}h_t`

With LSTM or GRU cells, the core principle is the same, but these type
of cells additionally use additional sigmoid units called gates that
allow to forget information or expose only particular part of the cell
state to the next step with a specified probability.

Sequence to Sequence Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The principle of Seq2Seq architecture is to encode information on an
input sequence :math:`x_1` to :math:`x_T` and to use this condensed
representation known as context vector to generate a new sequence
:math:`y_1` to :math:`y_{T'}`. Each output :math:`y_t` is based on
previous outputs and the context vector : > :math:`p(A|Q)` =
:math:`p(y_1, y_2,..., y_{T'} | x_1, x_2,..., x_T)` =
:math:`\prod_{t=1}^{T'} a_{i}` :math:`p(y_t | c_t, y_1,..., y_{t-1})`

The loss :math:`l` is the negative log-probability of the answer
sequence :math:`A = [y_1,...,y_T ]` given the question sequence
:math:`Q = [x_1,..., x_T]`, averaged over the number :math:`N` of
:math:`(Q,A)` pairs in a minibatch : > :math:`l` =
:math:`- \frac{1}{N} \sum_{(Q, A) \in N} p(A|Q)`

Multiple layers of RNN cells can be stacked over each other to increase
the model capacity like in a regular feedforward neural network.

The following figure presents a Seq2Seq model with a two layers encoder
and a two layers decoder:

.. figure:: ../docs/seq2seq.jpg
   :alt: Fig 1.SeqtoSeq Encoder-Decoder Architecture (Image Credit:https://github.com/tensorflow/nmt)

   alt text

This project uses a 512 cell RNN cell with two layers. Different weights
are used for encoder and decoder cells: >
:math:`W^{h*}_{encoder} \neq W^{h*}_{decoder}`

At the bottom layer, the encoder and decoder RNNs receive as input the
following: \* first, the source sentence, \* then a boundary marker
‘<GO>’ which indicates the transition from the encoding to the decoding
mode, and the target sentence.

For training, we will feed the system with the following tensors, which
are in time-major format and contain word indices: \* encoder_inputs
[max_encoder_sequence_length, batch_size]: source input words. The input
word array is reversed before feeding to the network. \* decoder_inputs
[max_decoder_sequence_length, batch_size]: target input words. \*
decoder_outputs [max_decoder_sequence_length, batch_size]: target output
words, these are decoder_inputs shifted to the left by one time step
with an end-of-sentence tag ‘<S>’ appended on the right.

Here for efficiency, we train with multiple sentences (batch_size = 30)
at one go. This is a hyper-parameter that is tuned while training.

Embeddings
''''''''''

Given the categorical nature of words, the model must first look up the
source and target embeddings to retrieve the corresponding word
representations. For this embedding layer to work, a vocabulary is first
chosen for each language. Usually, a vocabulary size V is selected, and
only the most frequent V words are treated as unique. All other words
are converted to an “unknown” token and all get the same embedding. The
embedding weights, one set per language, are usually learned during
training.

Note that one can choose to initialize embedding weights with pretrained
word representations such as word2vec or Glove vectors. In general,
given a large amount of training data we can learn these embeddings from
scratch, as we do in this project.

Encoder-Decoder
'''''''''''''''

Note that sentences have different lengths to avoid wasting computation,
we tell dynamic_rnn the max source sentence length for the batch through
sequence_length param and use padding to pad smaller sentences in the
batch. The encoder is trained with questions as the input while the
decoder is trained with answers as the input. The decoder also needs to
have access to the last encoder state information, and one simple way to
achieve that is to initialize it with the last hidden state of the
encoder.

Loss
''''

The output of the decoder layer is the unnormalized probabilities called
‘logits’. The loss is the cross entropy loss of the predicted output
with a expected output of class 1 of N. This sum of all logits is
divided by batch size to make hyper parameters invariant to batch size.

Optimization
''''''''''''

One of the important steps in training RNNs is gradient clipping. Here,
we clip by the global norm. The max value, max_gradient_norm, is often
set to a value like 5 or 1. We select Adam optimizer and a starting
learning rate in the range 0.0001 to 0.001; which decreases as training
progresses.

Attention Mechanism
^^^^^^^^^^^^^^^^^^^

The first drawback with the previous model is that all the relevant
information to decode is fed through a fixed size vector. When the
encoding sequence is long, it often fails to capture the complex
semantic relations between words entirely. On the other hand, if the
fixed size vector is too large for the encoding sequence, this may cause
overfitting problem forcing the encoder to learn some noise.

Furthermore, words occurring at the beginning of the encoding sentence
may contain information for predicting the first decoding outputs. It is
often complex for the model to capture long term dependencies, and there
is no guarantee that it will learn to handle these correctly. This
problem can be partially solved by using a Bi-Directional RNN (BD-RNN)
as the encoder. A BD-RNN will encode its input twice b y passing over
the input in both directions. We use a *Bidirectional Encoder* in this
project. This model allows for hidden representations at each timestep
to capture both future and past context information from the input
sequence.

The basic idea of attention is that instead of attempting to learn a
single vector representation for each sentence, we keep around vectors
for every word in the input sentence, and reference these vectors at
each decoding step.

.. figure:: ../docs/attention_mechanism.jpg
   :alt: Image credit: https://github.com/tensorflow/nmt

   alt text

It consists of the following stages: \* The current target hidden state
is compared with all source states to derive attention weights. \* Based
on the attention weights we compute a context vector as the weighted
average of the source states. \* Combine the context vector with the
current target hidden state to yield the final attention vector \* The
attention vector is fed as an input to the next time step *(input
feeding)*.

The first three steps can be summarized by the equations below: >
:math:`\alpha_{ts}` =
:math:`\frac{exp(score(h_t, \overline{h_s}))}{\sum_{r=1}^{S} exp(score(h_t, \overline{h_{r}}))}`

   :math:`c_t` = :math:`\sum_s \alpha_{ts}\overline{h_s}`

..

   :math:`a_t` = :math:`f(c_t, h_t)` = tanh(\ :math:`W_c[c_t; h_t]`)

Here, the function score is used to compared the target hidden state
:math:`h_t` with each of the source hidden states
:math:`\overline{h}_s`, and the result is normalized to produced
attention weights (a distribution over source positions). There are
various choices of the scoring function; popular scoring functions
include the multiplicative and additive forms. We use the additive form
(Bahdanau) using *tanh*. > :math:`score(h_t, \overline{h}_s)` =
:math:`v_a^T tanh(W_1h_t+W_2\overline{h}_s)`

Once computed, the attention vector :math:`a_t` is used to derive the
softmax logit and loss. This is similar to the target hidden state at
the top layer of a vanilla seq2seq model.

We also use a **dropout** with a keep probability of :math:`0.75`\ %

This model is used to train word embeddings and then further used to
train the reinforcement learning model using policy gradient method.

References
^^^^^^^^^^

-  Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015.\ `Neural
   machine translation by jointly learning to align and
   translate <https://arxiv.org/pdf/1409.0473.pdf>`__. ICLR.
-  Minh-Thang Luong, Hieu Pham, and Christopher D Manning.
   2015.\ `Effective approaches to attention-based neural machine
   translation <https://arxiv.org/pdf/1508.04025.pdf>`__. EMNLP.
-  Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. 2014.\ `Sequence to
   sequence learning with neural
   networks <https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf>`__.
   NIPS.
