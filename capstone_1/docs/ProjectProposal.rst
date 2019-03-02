
Deep learning for Sentiment Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Motivation
^^^^^^^^^^

*Sentiment analysis* is a field of study that analyzes
people's opinions, appraisals, attitudes, and emotions towards entities
such as products, services, organizations, individuals, issues, events,
topics, and their attributes.

Although linguistics and natural language processing (NLP) have a long
history, little research had been done about people’s *opinions* and
*sentiments*. Most current techniques (example, search engines work with
facts (example, knowledge graph) rather than working with opinions. The
advent of social media and availability of huge volumes of opinionated
data in these media have caused a resurgence in research in this field.

*Opinions* are key influencers of human behaviors. Businesses and
organizations always want to find consumer or public opinions about
their products and services. Individual consumers also want to know the
opinions of existing users of a product before purchasing it, and peer
opinions about political candidates before making a voting decision in a
political election. Examples of applications would be tracking user
opinions to predict stock prices for a company or predict box office
revenues for a movie.

Problem Description
^^^^^^^^^^^^^^^^^^^

This capstone project intends to explore recent advances in Natural
Language Processing to improve accuracy of sentiment classification. The
dataset used for the project would be the Rotten Tomatoes movie reviews
dataset. The Rotten Tomatoes movie review dataset is a corpus of movie
reviews used for sentiment analysis, originally collected by Pang and
Lee [1]. In their work on sentiment treebanks, Socher et al. [2] used
the Stanford parser [3] to create fine-grained labels for all parsed
phrases in the corpus annotated by three human judges. The Sentiment
Treebank along with the sentiment classification can be viewed at [5].

This is a **classification problem** where review phrases are labelled
on a scale of five values: *negative, somewhat negative, neutral,
somewhat positive, positive*. The goal of the project would be to
accurately classify the sentiment of a any movie review sentence.

In general, sentiment analysis is investigated at different
granularities, namely, *document, sentence* or *entity* levels. The
review text usually comprises of a single sentence in this dataset, and
the data does not contain which movie the review talked about. So, the
scope of the project is only restricted to sentence level analysis
without entity recognition as such. However, the text might be specific
to some aspect of the movie such as its screenplay. For example,

   *“The acting was mediocre, but the screenplay was top quality”.*

In this case, the screenplay had very positive review (emphasized) while
the acting had a very negative review (not emphasized). In general, every domain tends
to have a specific vocabulary that cannot be used for other domains. The
project scope is only restricted to movie review sentences.

Sentiment words themselves share a *common lexicon*. Example, would be
positive words such as great, excellent or amazing and negative words
such as bad, mediocre or terrible. Numerous efforts have been made to
build such lexicons, which are not sufficient as the problem is more
complex.

Some of the challenges with analyzing each review are:

-  **Language ambiguity**: Example,

::

   “The movie blows” vs “The movie blows away all expectations”.

..

    The word “blows” has negative orientation in one and positive in other.

-  **Contrapositive conjunction**: The review

::

   “The acting sucked but the movie was entertaining”.

..

   Here the review is of the form “X but Y”. The goal of the project would be to classify each phrase, X and Y accurately and then determine the overall sentiment for the movie, positive in this case.

-  **Sentence negation**: There are two kinds of examples here,

   - Negative positives:

::

    “The movie was not very great”.

..

   Here the phrase “very great” is positive but “not” changes the sentiment of the review.

   - Negative negatives:

::

   “The movie was not that terrible”.

..

   Here the phrase “terrible” is negative, but “not” does not make it positive.

-  **Sarcasm**: Sarcastic comments are hardest to detect and deal with.
   Example,

::

   “I was never this excited to leave the theater”.

..

   This is a very negative comment, but very hard to classify.

-  **Sentences with no opinions**: These are usually questions such as

::

   “Has any seen this movie?”

..

   or conditionals, such as,

::

   “If the acting is good, I will go and see the movie.”

..

   Both are neutral sentences. However not all questions or conditionals are neutral, example

::

   “Has anyone else seen this terrible movie?” or

   “If you looking for good acting, go and see this movie”.

-  **Sentences with no sentiment words but with opinions**: Example,

::

   “The movie solved my insomnia”.

..

   This is a very negative review without a sentiment word such as good or bad.

This project will evaluate the models only on first three of the issues
mentioned above. Language ambiguity is mitigated considerably by
restricting scope only to movie reviews. The other two issues,
contrapositive conjunction and sentence negation are mitigated by
constructing parse trees of the text and using compositionality
functions trained over known examples, which is what mostly the bulk of
this project is about. The sarcastic comments and the sentences with no
opinions will be not be treated differently than regular sentences. We expect the
tensor model to classify the cases with no sentiment words correctly as negative or
neutral respectively.

Methodology
^^^^^^^^^^^

The goal of the project is to build a sentiment classification system
using deep learning, namely Recursive Neural Tensor Network (RNTN). This
method uses tensors to remove dependency on the vocabulary and captures
different types of associations in the RNN.

The main components of the project would be:

-  **Training Engine**: Train the RNTN model with the training data set.

-  **Parser**: Parse the trees and extract sentiment labels.

-  **Prediction App**: A Web Application to view the predictions of any movie review.

-  **Stanford CoreNLP**: The project will reuse Stanford CoreNLP to do constituency
parsing of the sentence for which prediction needs to be made.

Data Set Description
^^^^^^^^^^^^^^^^^^^^

The project uses the data set from the original paper as contains fully
parsed trees and sentiment labels. The train, test and dev data already
split and parsed using the standard parser is exposed at
https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip

In addition, the original data set that the paper [2] uses the following
data: http://nlp.stanford.edu/sentiment/stanfordSentimentTreebank.zip
http://nlp.stanford.edu/sentiment/stanfordSentimentTreebankRaw.zip

The data contains raw scores in range (1 to 25) which are mapped to (1
to 5) range for both complete sentences and parsed sub phrases.