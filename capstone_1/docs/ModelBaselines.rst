
Baseline Models
~~~~~~~~~~~~~~~

We look at bag of words models as baseline models for comparison with
the RNTN model.

We evaluate the models for both root level and full tree node accuracy
scores.

Extracting Phrases from the Treebank
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Sentiment Treebank dataset is in form of parsed trees. Here we
generate all sub-phrases and their associated sentiments for evaluating
full accuracy.

.. code:: ipython3

    # Imports
    import os
    import sys
    import numpy as np
    import pandas as pd

.. code:: ipython3

    # Set path to model code
    PROJ_ROOT = os.pardir
    sys.path.append(PROJ_ROOT)
    from src.features.tree import Tree
    from src.models.data_manager import DataManager

.. code:: ipython3

    # Function to get sub-phrases for a single tree
    def get_phrases(node):
        if node.isLeaf:
            return (np.asarray([node.word]), np.asarray([node.label]))
        else:
            left_phrases, left_labels = get_phrases(node.left)
            right_phrases, right_labels = get_phrases(node.right)
            curr_phrases = np.concatenate([np.asarray([node.text()]), left_phrases, right_phrases])
            curr_labels = np.concatenate([np.asarray([node.label]), left_labels, right_labels])
            return (curr_phrases, curr_labels)

.. code:: ipython3

    # Get parsed trees
    trees_path = '../src/data/interim/trainDevTestTrees_PTB/trees/'
    x_train = DataManager(trees_path).x_train
    x_dev = DataManager(trees_path).x_dev
    x_test = DataManager(trees_path).x_test

.. code:: ipython3

    # Get sub-phrases for every tree
    X_data_train = []
    y_data_train = []
    for i in range(len(x_train)):
        X_tree, y_tree = get_phrases(x_train[i].root)
        X_data_train = np.concatenate([X_data_train, X_tree])
        y_data_train = np.concatenate([y_data_train, y_tree])
        
    dt_train = pd.DataFrame(data={'phrase': X_data_train, 'label': y_data_train})
    dt_train.to_csv('../src/data/processed/train_phrases.csv')

.. code:: ipython3

    # or run only the following
    dt_train = pd.read_csv('../src/data/processed/train_phrases.csv')
    X_data_train = np.ravel(dt_train[['phrase']])
    y_data_train = np.ravel(dt_train[['label']])

.. code:: ipython3

    # Get sub-phrases for every cross validation set tree
    X_data_dev = []
    y_data_dev = []
    for i in range(len(x_dev)):
        X_tree, y_tree = get_phrases(x_dev[i].root)
        X_data_dev = np.concatenate([X_data_dev, X_tree])
        y_data_dev = np.concatenate([y_data_dev, y_tree])
    
    dt_dev = pd.DataFrame(data={'phrase': X_data_dev, 'label': y_data_dev})
    dt_dev.to_csv('../src/data/processed/dev_phrases.csv')

.. code:: ipython3

    # or run only the following
    dt_dev = pd.read_csv('../src/data/processed/dev_phrases.csv')
    X_data_dev = np.ravel(dt_dev[['phrase']])
    y_data_dev = np.ravel(dt_dev[['label']])

Building vocabulary
^^^^^^^^^^^^^^^^^^^

The vocabulary is built using a CountVectorizer that extracts words and
pre-processes them (lemmatization). The fit_transform method returns the
one-hot encoded version of the sentences with the frequency of the word
occurence as the components of the generated sentence vector (rows).

.. code:: ipython3

    # Build vocabulary using CountVectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    
    vectorizer = CountVectorizer()
    X_data = vectorizer.fit_transform(np.concatenate([X_data_train, X_data_dev]))
    X_data = X_data.tocsc()  # some versions of sklearn return COO format
    y_data = np.concatenate([y_data_train, y_data_dev])

Cross validation
^^^^^^^^^^^^^^^^

The dev/train split is already specified in the trained dataset. Here we
use Predefined Split to specify which data is cross-validation test set
and which is training data.

.. code:: ipython3

    # Use Predefined split as train, dev data is already separate
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.model_selection import PredefinedSplit, GridSearchCV
    
    # Prepare data for training
    validation_set_indexes = [-1] * len(X_data_train) + [0] * len(X_data_dev)
    cv = PredefinedSplit(test_fold=validation_set_indexes)

Naive Bayes Model
^^^^^^^^^^^^^^^^^

The Naive Bayes model provides a good baseline as it only makes
independence assumption. However, we do not expect it to do well against
sentences with negation as it does not take structure of the sentence in
account. It will also mark sentences with higher positive word counts
more positively and similarly negative word counts will give negative
sentiment as the prediction.

.. code:: ipython3

    # Simple naive bayes classifier
    from sklearn.metrics import make_scorer, accuracy_score, f1_score, log_loss
    
    # Use MultinomialNB classifier
    from sklearn.naive_bayes import MultinomialNB
    nb_clf = MultinomialNB()
    
    # Find the best hyper-parameter using GridSearchCV
    params = {'alpha': [.1, 1, 10]}
    nb_model = GridSearchCV(nb_clf, params, scoring=make_scorer(accuracy_score), cv=cv)

.. code:: ipython3

    # Train model
    nb_model.fit(X_data, y_data)
    print(nb_model.best_params_)


.. parsed-literal::

    {'alpha': 1}
    

We load the test dataset to compare. Phrases are extracted out of each
sentence as we know their sentiment labels.

.. code:: ipython3

    # Get sub-phrases for every test set tree
    X_data_test = []
    y_data_test = []
    for i in range(len(x_test)):
        X_tree, y_tree = get_phrases(x_test[i].root)
        X_data_test = np.concatenate([X_data_test, X_tree])
        y_data_test = np.concatenate([y_data_test, y_tree])
        
    dt_test = pd.DataFrame(data={'phrase': X_data_test, 'label': y_data_test})
    dt_test.to_csv('../src/data/processed/test_phrases.csv')

.. code:: ipython3

    # or run only the following
    dt_test = pd.read_csv('../src/data/processed/test_phrases.csv')
    X_data_test = np.ravel(dt_test[['phrase']])
    y_data_test = np.ravel(dt_test[['label']])

Generate a word frequency count vector for each sentence.

.. code:: ipython3

    # Vectorize
    X_data_test_vec = vectorizer.transform(X_data_test)

.. code:: ipython3

    # Score model
    # Print the accuracy on the test and training dataset
    #training_accuracy = model.score(X_data.reshape(-1,1), y_data)
    #test_accuracy = model.score(X_data_test_vec, y_data_test.astype(int))
    y_pred = nb_model.predict(X_data_test_vec)
    y_true = y_data_test.astype(int)
    y_probs = nb_model.predict_proba(X_data_test_vec)
    
    #print("Accuracy on training data: {:2f}".format(training_accuracy))
    print("Accuracy on full test data (NB):     {:2f}".format(accuracy_score(y_true, y_pred)))


.. parsed-literal::

    Accuracy on full test data (NB):     0.735557
    

The model gives a good accuracy score. Next we look at where the
misclassifications are happening.

.. code:: ipython3

    # Classification Report
    print(classification_report(y_true, y_pred))


.. parsed-literal::

                  precision    recall  f1-score   support
    
               0       0.44      0.09      0.15      2008
               1       0.57      0.19      0.29      9255
               2       0.76      0.97      0.86     56548
               3       0.54      0.29      0.38     10998
               4       0.60      0.17      0.26      3791
    
       micro avg       0.74      0.74      0.74     82600
       macro avg       0.58      0.34      0.39     82600
    weighted avg       0.70      0.74      0.68     82600
    
    

.. code:: ipython3

    # F1-score
    print(f1_score(y_true, y_pred, average='weighted'))


.. parsed-literal::

    0.6833294980935176
    

.. code:: ipython3

    # Confusion Matrix
    print(confusion_matrix(y_true, y_pred))


.. parsed-literal::

    [[  177   606  1170    52     3]
     [  168  1779  7030   250    28]
     [   50   595 54967   856    80]
     [    7   131  7347  3194   319]
     [    0    29  1506  1616   640]]
    

The misclassifications show where the problem lies. The class imbalance
is causing the classifier to make more ‘neutral’ sentiment predictions.
Even for more extreme values, the classifications error towards neutral
state.

The weighted F1 Score gives a good overall measure to directly evaluate
the models, the other is log loss as shown below.

.. code:: ipython3

    # Log loss per sample
    print(log_loss(y_true, y_probs))


.. parsed-literal::

    0.8609283543125711
    

Fixing imbalance with oversampling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To reduce misclassification due to imbalance, we try oversampling the
minority classes. Undersampling would not be a good choice as the
vocabulary matrix is sparse and this will result in model classifying
most of the 1-grams not seen as neutral.

.. code:: ipython3

    from imblearn.over_sampling import RandomOverSampler
    from collections import Counter
    
    ros = RandomOverSampler(random_state=42)
    X_os, y_os = ros.fit_resample(X_data_test_vec, y_data_test.astype(int))
    
    print('After Rebalance: {0}'.format(Counter(y_os)))


.. parsed-literal::

    After Rebalance: Counter({2: 56548, 3: 56548, 1: 56548, 4: 56548, 0: 56548})
    

.. code:: ipython3

    # Train model
    nb_rebal_model = MultinomialNB()
    nb_rebal_model.fit(X_os, y_os)
    
    # Score model after rebalance
    y_pred = nb_rebal_model.predict(X_data_test_vec)
    y_true = y_data_test.astype(int)
    y_probs = nb_rebal_model.predict_proba(X_data_test_vec)

.. code:: ipython3

    #print("Accuracy on training data: {:2f}".format(training_accuracy))
    print("Accuracy on full test data (NB):     {:2f}".format(accuracy_score(y_true, y_pred)))


.. parsed-literal::

    Accuracy on full test data (NB):     0.338172
    

Oversampling does not improve the accuracy of the classification at all.
We will use the original unbalanced sample as the baseline.

Root Level Evaluation Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next we examine how the root level accuracy and classification metrics
are, which gives us the overall prediction for a sentence.

.. code:: ipython3

    # Build root train data set
    x_root_train = [tree.text() for tree in x_train]
    y_root_train = [tree.root.label for tree in x_train]
    x_root_dev = [tree.text() for tree in x_dev]
    y_root_dev = [tree.root.label for tree in x_dev]
    x_root_all = x_root_train + x_root_dev
    y_root_all = y_root_train + y_root_dev

.. code:: ipython3

    # Vectorize x
    x_root_train_vec = vectorizer.transform(x_root_all)

.. code:: ipython3

    # Train model for root nodes
    nb_root = MultinomialNB()
    nb_root.fit(x_root_train_vec, y_root_all)




.. parsed-literal::

    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)



.. code:: ipython3

    # Build root test data set
    x_root_test = [tree.text() for tree in x_test]
    y_root_test = [tree.root.label for tree in x_test]
    x_root_test_vec = vectorizer.transform(x_root_test)

.. code:: ipython3

    # Score model
    # Print the accuracy on the test dataset
    y_pred = nb_model.predict(x_root_test_vec)
    y_true = y_root_test
    y_probs = nb_model.predict_proba(x_root_test_vec)
    
    print("Accuracy on root test data (NB):     {:2f}".format(accuracy_score(y_true, y_pred)))


.. parsed-literal::

    Accuracy on root test data (NB):     0.319457
    

.. code:: ipython3

    # Classification Report
    print(classification_report(y_true, y_pred))


.. parsed-literal::

                  precision    recall  f1-score   support
    
               0       0.65      0.11      0.18       279
               1       0.51      0.23      0.32       633
               2       0.21      0.75      0.33       389
               3       0.40      0.32      0.36       510
               4       0.66      0.19      0.30       399
    
       micro avg       0.32      0.32      0.32      2210
       macro avg       0.49      0.32      0.30      2210
    weighted avg       0.48      0.32      0.31      2210
    
    

.. code:: ipython3

    # F1-score
    print(f1_score(y_true, y_pred, average='weighted'))


.. parsed-literal::

    0.3088168253391177
    

.. code:: ipython3

    # Confusion Matrix
    print(confusion_matrix(y_true, y_pred))


.. parsed-literal::

    [[ 30  78 165   5   1]
     [ 13 147 435  33   5]
     [  3  37 290  54   5]
     [  0  19 300 163  28]
     [  0   5 165 153  76]]
    

.. code:: ipython3

    # Log loss per sample
    print(log_loss(y_true, y_probs))


.. parsed-literal::

    2.4381252006114305
    

The root level accuracy is much lower as expected. This also aligns with
a max accuracy of about 45% for the best model in the paper.

Here, too we see distinct effect of too many neutral words on the
overall accuracy of the model.