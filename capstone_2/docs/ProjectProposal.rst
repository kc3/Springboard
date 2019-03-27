
Deep Reinforcement Learning for Dialogue Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Motivation
^^^^^^^^^^

Conversational bots are one of the most exciting applications of NLP.
Services such as Google Assistant, Apple Siri and Amazon Alexa are
becoming ubiquitous and improving in capabilities rapidly. Chatbots are
typically used in dialog systems for various practical purposes
including customer service or information acquisition. Some chatbots use
sophisticated natural language processing systems, but many simpler ones
scan for keywords within the input, then pull a reply with the most
matching keywords, or the most similar wording pattern, from a database.

Problem Description
^^^^^^^^^^^^^^^^^^^

Efforts to build statistical dialog systems fall into two major
categories, statistical machine translation models and task-oriented
dialog systems.

The first treats dialogue generation as a source to target transduction
problem and learns mapping rules between input messages and responses
from a massive amount of training data. Seq2Seq models [2] or their
variants are used for generating responses.

Seq2Seq model works by generating an internal representation of the
(sentence, response) pairs in the dataset using a Recursive Neural
Network using auto-encoders and then building a decoder to decode the
internal representation into a sequence of words that would be the most
likely response. The problem faced with this model is a high generation
of “I don’t know” responses, which is a direct consequence of such
occurrences in training text.

The second method focus on solving domain-specific tasks and often rely
on carefully limited dialogue parameters, or hand-built templates with
state, action and reward signals designed by humans for each new domain,
making the paradigm difficult to extend to open-domain scenarios.

There are three problems that conversational dialog systems typically
have:

-  **Dull and generic responses**: Example,

..

   A: *How is life?*

   B: *I don’t know what you are talking about.*

Another example of this is:

   A: *How old are you?*

..

   B: *I don’t know.*

This can be explained as most systems try to maximize the probability of
the next response, and *I don’t know.* is an adequate response for most
cases. This is further aggravated by the fact that this response shows
up a lot in the training data set.

-  **Repetitive Responses**: Example,

..

   A: See you later!

   B: See you later.

..

   A: See you later.

and so on. Another example is,

   A: Good bye!

..

   B: Good bye!

   A: Good bye!

and so on.

-  **Short-sighted responses**: These systems tend to pick responses
   that do not increase the length of the conversation or make it
   interesting or even coherent. An example,

..

   A: How old are you?

   B: I am 16.

..

   A: 16?

   B: I don’t know what you are talking about?

..

   A: You don’t know what you are saying?

and so on. The first response by agent B is a very close ended response
and it is very hard for the agent A to come up with a good response for
this. Unfortunately, the results of the response ‘*I am 16*’ does not
show up a few turns later. This is the main motivation for using
Reinforcement Learning to solve this problem.

Methodology
^^^^^^^^^^^

The goal of this project would be to build a bot that can have an open
dialogue with the user, having three important conversational
properties: informativity, coherence and ease of answering.

This project solves the problem by creating a learning system of two
agents already trained with supervised seq2seq model and each talking
with one another with a goal of maximizing long term rewards, such as
increasing the length of conversation and metrics mentioned above.

The main components of the project would be:

-  **Training Engine**: An implementation of a LSTM based attention
   model for this project.
-  **Reinforcement Engine**: An implementation of policy gradient method
   for training the dataset.
-  **Chat App**: A web application to have conversations with the
   chatbot.

Dataset Description
^^^^^^^^^^^^^^^^^^^

The original paper uses the OpenSubtitles dataset. For this project we
will be using the Cornell Movie Dialogs Corpus instead, partly because
it has meta-data and it is of smaller size.

This corpus contains a large metadata-rich collection of fictional
conversations extracted from raw movie scripts:

-  220,579 conversational exchanges between 10,292 pairs of movie
   characters

-  involves 9,035 characters from 617 movies

-  in total 304,713 utterances

-  movie metadata included:

   -  genres

   -  release year

   -  IMDB rating

   -  number of IMDB votes

   -  IMDB rating

-  character metadata included:

   -  gender (for 3,774 characters)

   -  position on movie credits (3,321 characters)

Dataset Location:
https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

References
^^^^^^^^^^

[1] Deep Reinforcement Learning for Dialogue Generation, Li et al

[2] Sequence to Sequence Learning with Neural Networks - Sutskever et
al.
