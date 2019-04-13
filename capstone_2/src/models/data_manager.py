
import os
import re
import joblib
import logging


class DataManager:
    """Type to handle data manipulation for this project."""

    _abs_path = os.path.abspath(os.path.dirname(__file__))

    _def_cornell_path = os.path.join(_abs_path, '../data/interim')

    _def_models_path = os.path.join(_abs_path, '../../models')

    _def_processed_path = os.path.join(_abs_path, '../data/processed')

    __instance = None

    def __new__(cls, *args, **kwargs):
        """Create the object on first instantiation."""

        if cls.__instance is None:
            cls.__instance = object.__new__(cls)

        return cls.__instance

    def __init__(self):
        self.contractions_dict = {
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
        self.contractions_re = re.compile('(%s)' % '|'.join(self.contractions_dict.keys()), re.IGNORECASE)
        self.sorted_questions, self.sorted_answers, self.questions_int_to_vocab, self.answers_int_to_vocab = \
            self.get_cornell_data()
        self.questions_vocab_to_int = {v_i: v for v, v_i in self.questions_int_to_vocab.items()}
        self.answers_vocab_to_int = {v_i: v for v, v_i in self.answers_int_to_vocab.items()}

    def expand_contractions(self, s):
        def replace(match):
            return self.contractions_dict[match.group(0).lower()]

        text = self.contractions_re.sub(replace, s)
        return re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)

    def get_cornell_data(self):

        # Check if data exists
        if self._check_cornell_data():
            return self._load_cornell_data()

        # Load the data
        file_movie_lines = '{0}/movie_lines.txt'.format(self._def_cornell_path)
        lines = open(file_movie_lines, encoding='utf-8', errors='ignore').read().split('\n')

        file_movie_convs = '{0}/movie_conversations.txt'.format(self._def_cornell_path)
        conv_lines = open(file_movie_convs, encoding='utf-8', errors='ignore').read().split('\n')

        # Create a dictionary to map each line's id with its text
        id2line = {}
        for line in lines:
            _line = line.split(' +++$+++ ')
            if len(_line) == 5:
                id2line[_line[0]] = _line[4]

        # Add the sentence end marker
        id2line['L0'] = '<EOC>'

        # Create a list of all of the conversations' lines' ids.
        convs = []
        for line in conv_lines[:-1]:
            _line = line.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
            convs.append(_line.split(','))

        # Sort the sentences into questions (inputs) and answers (targets)
        questions = []
        answers = []

        for conv in convs:
            for i in range(len(conv) - 1):
                questions.append(id2line[conv[i]])
                answers.append(id2line[conv[i + 1]])

            # Add a conversation end marker
            questions.append(id2line[conv[len(conv) - 1]])
            answers.append(id2line['L0'])

        # Clean the data
        clean_questions = []
        for question in questions:
            clean_questions.append(self.expand_contractions(question))

        clean_answers = []
        for answer in answers:
            if answer != '<EOC>':
                clean_answers.append(self.expand_contractions(answer))
            else:
                clean_answers.append(answer)

        # Remove questions and answers that are shorter than 2 words and longer than 20 words.
        min_line_length = 2
        max_line_length = 20

        # Filter out the questions that are too short/long
        short_questions_temp = []
        short_answers_temp = []

        i = 0
        for question in clean_questions:
            if min_line_length <= len(question.split()) <= max_line_length:
                short_questions_temp.append(question)
                short_answers_temp.append(clean_answers[i])
            i += 1

        # Filter out the answers that are too short/long
        short_questions = []
        short_answers = []

        i = 0
        for answer in short_answers_temp:
            if min_line_length <= len(answer.split()) <= max_line_length:
                short_answers.append(answer)
                short_questions.append(short_questions_temp[i])
            i += 1

        # Create a dictionary for the frequency of the vocabulary
        vocab = {}
        for question in short_questions:
            for word in question.split():
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1

        for answer in short_answers:
            for word in answer.split():
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1

        # Remove rare words from the vocabulary.
        # We will aim to replace fewer than 5% of words with <UNK>
        threshold = 10
        count = 0
        for k, v in vocab.items():
            if v >= threshold:
                count += 1

        # In case we want to use a different vocabulary sizes for the source and target text,
        # we can set different threshold values.
        # Nonetheless, we will create dictionaries to provide a unique integer for each word.
        questions_vocab_to_int = {}

        word_num = 0
        for word, count in vocab.items():
            if count >= threshold:
                questions_vocab_to_int[word] = word_num
                word_num += 1

        answers_vocab_to_int = {}

        word_num = 0
        for word, count in vocab.items():
            if count >= threshold:
                answers_vocab_to_int[word] = word_num
                word_num += 1

        # Add the unique tokens to the vocabulary dictionaries.
        codes = ['<PAD>', '<EOS>', '<UNK>', '<GO>']

        for code in codes:
            questions_vocab_to_int[code] = len(questions_vocab_to_int)

        for code in codes:
            answers_vocab_to_int[code] = len(answers_vocab_to_int)

        # Create dictionaries to map the unique integers to their respective words.
        # i.e. an inverse dictionary for vocab_to_int.
        questions_int_to_vocab = {v_i: v for v, v_i in questions_vocab_to_int.items()}
        answers_int_to_vocab = {v_i: v for v, v_i in answers_vocab_to_int.items()}

        # Add the end of sentence token to the end of every answer.
        for i in range(len(short_answers)):
            short_answers[i] += ' <EOS>'

        # Convert the text to integers.
        # Replace any words that are not in the respective vocabulary with <UNK>
        questions_int = []
        for question in short_questions:
            ints = []
            for word in question.split():
                if word not in questions_vocab_to_int:
                    ints.append(questions_vocab_to_int['<UNK>'])
                else:
                    ints.append(questions_vocab_to_int[word])
            questions_int.append(ints)

        answers_int = []
        for answer in short_answers:
            ints = []
            for word in answer.split():
                if word not in answers_vocab_to_int:
                    ints.append(answers_vocab_to_int['<UNK>'])
                else:
                    ints.append(answers_vocab_to_int[word])
            answers_int.append(ints)

        # Sort questions and answers by the length of questions.
        # This will reduce the amount of padding during training
        # Which should speed up training and help to reduce the loss

        sorted_questions = []
        sorted_answers = []

        for length in range(1, max_line_length + 1):
            for i in enumerate(questions_int):
                if len(i[1]) == length:
                    sorted_questions.append(questions_int[i[0]])
                    sorted_answers.append(answers_int[i[0]])

        # Save the files
        self._save_cornell_data(sorted_questions, sorted_answers, questions_int_to_vocab, answers_int_to_vocab)

        return sorted_questions, sorted_answers, questions_int_to_vocab, answers_int_to_vocab

    def _save_cornell_data(self, sorted_questions, sorted_answers, questions_int_to_vocab, answers_int_to_vocab):
        """Pickles files to processed folder"""

        file_path = '{0}/sorted_questions.pkl'.format(self._def_processed_path)
        joblib.dump(sorted_questions, file_path, compress=9)

        file_path = '{0}/sorted_answers.pkl'.format(self._def_processed_path)
        joblib.dump(sorted_answers, file_path, compress=9)

        file_path = '{0}/questions_int_to_vocab.pkl'.format(self._def_processed_path)
        joblib.dump(questions_int_to_vocab, file_path, compress=9)

        file_path = '{0}/answers_int_to_vocab.pkl'.format(self._def_processed_path)
        joblib.dump(answers_int_to_vocab, file_path, compress=9)
        logging.info('Saved Cornell Data to processed folder.')

    def _load_cornell_data(self):
        """Loads Cornell data from pickled files."""

        file_path = '{0}/sorted_questions.pkl'.format(self._def_processed_path)
        sorted_questions = joblib.load(file_path)

        file_path = '{0}/sorted_answers.pkl'.format(self._def_processed_path)
        sorted_answers = joblib.load(file_path)

        file_path = '{0}/questions_int_to_vocab.pkl'.format(self._def_processed_path)
        questions_int_to_vocab = joblib.load(file_path)

        file_path = '{0}/answers_int_to_vocab.pkl'.format(self._def_processed_path)
        answers_int_to_vocab = joblib.load(file_path)

        logging.info('Loaded Cornell Data from processed folder.')
        return sorted_questions, sorted_answers, questions_int_to_vocab, answers_int_to_vocab

    def _check_cornell_data(self):
        """Checks if pickled data exists"""
        file_path = '{0}/sorted_questions.pkl'.format(self._def_processed_path)
        return os.path.exists(file_path)

    @staticmethod
    def _load(file_path, max_rows=None):
        """Loads entire content of the file."""
        s = []
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if max_rows is None or i < max_rows:
                    t = line.strip()
                    s.append(t)
                else:
                    break

            return s

    def question_to_tokens(self, question):
        """Converts text to tokens in vocabulary."""
        # Clean question
        question = self.expand_contractions(question)

        # Load dict for word -> int
        file_path = '{0}/questions_int_to_vocab.pkl'.format(self._def_processed_path)
        questions_int_to_vocab = joblib.load(file_path)
        questions_vocab_to_int = {v_i: v for v, v_i in questions_int_to_vocab.items()}

        # Convert text to ints
        tokens = []
        for word in question.split():
            if word not in questions_vocab_to_int:
                tokens.append(questions_vocab_to_int['<UNK>'])
            else:
                tokens.append(questions_vocab_to_int[word])

        return tokens

    def answer_from_tokens(self, answer):
        """Converts vocabulary tokens to text."""

        words = []
        for token in answer:
            if token in self.answers_int_to_vocab:
                words.append(self.answers_int_to_vocab[token])
            else:
                words.append(self.answers_int_to_vocab['<UNK>'])

        sentence = ' '.join(words)

        return sentence

    def get_cornell_starting_prompts(self):

        questions = [
            "How about dinner Saturday night?",
            "There is a fight going on in the quad.",
            "What do we do about this project?"
        ]

        q_tokens = []

        for question in questions:
            q_clean = self.expand_contractions(question)
            q_tokens.append(self.question_to_tokens(q_clean))

        return q_tokens

    def get_cornell_dull_responses(self):

        questions = [
            "I am not <UNK>",
            "I do not know that",
            "I am not sure",
            "I do not think that",
            "I am sorry"
        ]

        q_tokens = []

        for question in questions:
            q_clean = self.expand_contractions(question)
            q_tokens.append(self.question_to_tokens(q_clean))

        return q_tokens

