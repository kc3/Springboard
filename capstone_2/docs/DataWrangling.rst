
Data Wrangling
~~~~~~~~~~~~~~

Data Description
^^^^^^^^^^^^^^^^

The Cornell Movie Dialog corpus contains a metadata-rich collection of
fictional conversations extracted from raw movie scripts:

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

B) Files description:

In all files the field separator is " +++$+++ "

-  movie_titles_metadata.txt

   -  contains information about each movie title
   -  fields:

      -  movieID,
      -  movie title,
      -  movie year,
      -  IMDB rating,
      -  no. IMDB votes,
      -  genres in the format [‘genre1’,‘genre2’,..,‘genreN’]

-  movie_characters_metadata.txt

   -  contains information about each movie character
   -  fields:

      -  characterID
      -  character name
      -  movieID
      -  movie title
      -  gender (“?” for unlabeled cases)
      -  position in credits (“?” for unlabeled cases)

-  movie_lines.txt

   -  contains the actual text of each utterance
   -  fields:

      -  lineID
      -  characterID (who uttered this phrase)
      -  movieID
      -  character name
      -  text of the utterance

-  movie_conversations.txt

   -  the structure of the conversations
   -  fields

      -  characterID of the first character involved in the conversation
      -  characterID of the second character involved in the
         conversation
      -  movieID of the movie in which the conversation occurred
      -  list of the utterances that make the conversation, in
         chronological order: [‘lineID1’,‘lineID2’,..,‘lineIDN’] has to
         be matched with movie_lines.txt to reconstruct the actual
         content

-  raw_script_urls.txt

   -  the urls from which the raw sources were retrieved

Load Data
^^^^^^^^^

We are primarily going to use movie_conversations.txt and
movie_lines.txt. First load the movie_lines.txt and extract the lines.

.. code:: ipython3

    # Python imports
    import numpy as np
    import pandas as pd
    import re

.. code:: ipython3

    # Load the data
    lines = open('../src/data/interim/movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
    conv_lines = open('../src/data/interim/movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

.. code:: ipython3

    # The sentences that we will be using to train our model.
    lines[:10]




.. parsed-literal::

    ['L1045 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ They do not!',
     'L1044 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ They do to!',
     'L985 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ I hope so.',
     'L984 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ She okay?',
     "L925 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ Let's go.",
     'L924 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ Wow',
     "L872 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ Okay -- you're gonna need to learn how to lie.",
     'L871 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ No',
     'L870 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ I\'m kidding.  You know how sometimes you just become this "persona"?  And you don\'t know how to quit?',
     'L869 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ Like my fear of wearing pastels?']



.. code:: ipython3

    # The sentences' ids, which will be processed to become our input and target data.
    conv_lines[:10]




.. parsed-literal::

    ["u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197']",
     "u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L198', 'L199']",
     "u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L200', 'L201', 'L202', 'L203']",
     "u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L204', 'L205', 'L206']",
     "u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L207', 'L208']",
     "u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L271', 'L272', 'L273', 'L274', 'L275']",
     "u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L276', 'L277']",
     "u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L280', 'L281']",
     "u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L363', 'L364']",
     "u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L365', 'L366']"]



.. code:: ipython3

    # Create a dictionary to map each line's id with its text
    id2line = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]

.. code:: ipython3

    # Add the sentence end marker
    id2line['L0'] = '<EOC>'

.. code:: ipython3

    # Create a list of all of the conversations' lines' ids.
    convs = [ ]
    for line in conv_lines[:-1]:
        _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
        convs.append(_line.split(','))

.. code:: ipython3

    convs[:10]




.. parsed-literal::

    [['L194', 'L195', 'L196', 'L197'],
     ['L198', 'L199'],
     ['L200', 'L201', 'L202', 'L203'],
     ['L204', 'L205', 'L206'],
     ['L207', 'L208'],
     ['L271', 'L272', 'L273', 'L274', 'L275'],
     ['L276', 'L277'],
     ['L280', 'L281'],
     ['L363', 'L364'],
     ['L365', 'L366']]



.. code:: ipython3

    # Sort the sentences into questions (inputs) and answers (targets)
    questions = []
    answers = []
    
    for conv in convs:
        for i in range(len(conv)-1):
            questions.append(id2line[conv[i]])
            answers.append(id2line[conv[i+1]])
            
        # Add a conversation end marker
        questions.append(id2line[conv[len(conv)-1]])
        answers.append(id2line['L0'])

.. code:: ipython3

    # Check if we have loaded the data correctly
    limit = 0
    for i in range(limit, limit+5):
        print(questions[i])
        print(answers[i])
        print()


.. parsed-literal::

    Can we make this quick?  Roxanne Korrine and Andrew Barrett are having an incredibly horrendous public break- up on the quad.  Again.
    Well, I thought we'd start with pronunciation, if that's okay with you.
    
    Well, I thought we'd start with pronunciation, if that's okay with you.
    Not the hacking and gagging and spitting part.  Please.
    
    Not the hacking and gagging and spitting part.  Please.
    Okay... then how 'bout we try out some French cuisine.  Saturday?  Night?
    
    Okay... then how 'bout we try out some French cuisine.  Saturday?  Night?
    <EOC>
    
    You're asking me out.  That's so cute. What's your name again?
    Forget it.
    
    

Data Cleaning
^^^^^^^^^^^^^

Next we work on cleaning data by expanding English contractions such as
“don’t” for “do not”.

.. code:: ipython3

    contractions_dict = { 
        "ain't": "am not ",
        "aren't": "are not",
        "'bout": "about",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "I had",
        "i'd've": "I would have",
        "i'll": "I will",
        "i'll've": "I will have",
        "i'm": "I am",
        "i've": "I have",
        "isn't": "is not",
        "it'd": "it had",
        "ot'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so is",
        "that'd": "that had",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you had",
        "you'd've": "you would have",
        "you'll": "you shall",
        "you'll've": "you shall have",
        "you're": "you are",
        "you've": "you have"
    }
    
    contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()), re.IGNORECASE)
    
    def expand_contractions(s, contractions_dict=contractions_dict):
        def replace(match):
            return contractions_dict[match.group(0).lower()]
        text = contractions_re.sub(replace, s)
        return re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    
    
    expand_contractions('you\'re great!')




.. parsed-literal::

    'you are great'



.. code:: ipython3

    # Clean the data
    clean_questions = []
    for question in questions:
        clean_questions.append(expand_contractions(question))
        
    clean_answers = []    
    for answer in answers:
        if answer != '<EOC>':
            clean_answers.append(expand_contractions(answer))
        else:
            clean_answers.append(answer)

.. code:: ipython3

    # Take a look at some of the data to ensure that it has been cleaned well.
    limit = 0
    for i in range(limit, limit+5):
        print(clean_questions[i])
        print(clean_answers[i])
        print()


.. parsed-literal::

    Can we make this quick  Roxanne Korrine and Andrew Barrett are having an incredibly horrendous public break up on the quad  Again
    Well I thought we would start with pronunciation if that is okay with you
    
    Well I thought we would start with pronunciation if that is okay with you
    Not the hacking and gagging and spitting part  Please
    
    Not the hacking and gagging and spitting part  Please
    Okay then how about we try out some French cuisine  Saturday  Night
    
    Okay then how about we try out some French cuisine  Saturday  Night
    <EOC>
    
    you are asking me out  that is so cute what is your name again
    Forget it
    
    

We now have a clean Question and Answer datasets that are used for
further processing and gathering further insights.

.. code:: ipython3

    # Save the files
    conv_final = pd.DataFrame({'question': clean_questions, 'answer': clean_answers})
    conv_final.to_csv('../src/data/processed/movie_qa.txt')

.. code:: ipython3

    conv_final.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>question</th>
          <th>answer</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Can we make this quick  Roxanne Korrine and An...</td>
          <td>Well I thought we would start with pronunciati...</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Well I thought we would start with pronunciati...</td>
          <td>Not the hacking and gagging and spitting part ...</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Not the hacking and gagging and spitting part ...</td>
          <td>Okay then how about we try out some French cui...</td>
        </tr>
        <tr>
          <th>3</th>
          <td>Okay then how about we try out some French cui...</td>
          <td>&lt;EOC&gt;</td>
        </tr>
        <tr>
          <th>4</th>
          <td>you are asking me out  that is so cute what is...</td>
          <td>Forget it</td>
        </tr>
      </tbody>
    </table>
    </div>


