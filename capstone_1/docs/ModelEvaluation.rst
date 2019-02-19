
Model Evaluation
~~~~~~~~~~~~~~~~

We first look at accuracy metrics and try to identify what is
mis-classified most.

Evaluation Setup
^^^^^^^^^^^^^^^^

Load the model and test data.

.. code:: ipython3

    # Imports
    import os
    import sys
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss

.. code:: ipython3

    # Add src to path
    PROJ_ROOT = os.pardir
    sys.path.append(PROJ_ROOT)
    from src.features.tree import Tree
    from src.models.data_manager import DataManager
    from src.models.rntn import RNTN
    from src.features.tree import Tree

.. code:: ipython3

    # Best Model 
    model_name = 'RNTN_30_tanh_35_5_None_50_0.001_0.01_9645'
    
    # Instantiate model
    model_rntn = RNTN(model_name=model_name)

.. code:: ipython3

    # Load test data for full tree
    x_test = DataManager().x_test

.. code:: ipython3

    # Function to get sub-phrases for a single tree
    def get_phrases(node):
        if node.isLeaf:
            return (np.asarray([str(node)]), np.asarray([node.label]))
        else:
            left_phrases, left_labels = get_phrases(node.left)
            right_phrases, right_labels = get_phrases(node.right)
            curr_phrases = np.concatenate([np.asarray([str(node)]), left_phrases, right_phrases])
            curr_labels = np.concatenate([np.asarray([node.label]), left_labels, right_labels])
            return (curr_phrases, curr_labels)
    
    X_data = []
    y_data = []
    
    for i in range(len(x_test)):
        X_tree, y_tree = get_phrases(x_test[i].root)
        X_data = np.concatenate([X_data, X_tree])
        y_data = np.concatenate([y_data, y_tree])
    
    dt_test = pd.DataFrame(data={'phrase': X_data})
    dt_test.to_csv('../src/data/processed/test_phrases_raw.csv')

.. code:: ipython3

    X_trees_data = [Tree(t) for t in X_data]

Full Tree Accuracy
^^^^^^^^^^^^^^^^^^

The test data contains the each sentence and its sub-phrase and
associated ground truth label. We use the model predict function to look
at how each node is predicted.

.. code:: ipython3

    # Call models predict method
    y_pred = model_rntn.predict(np.asarray(X_trees_data).reshape(-1, 1))
    y_true = y_data.astype(int)


.. parsed-literal::

    INFO:tensorflow:Restoring parameters from C:\Users\cskap\github\Springboard\capstone_1\src\models\../../models//RNTN_30_tanh_35_5_None_50_0.001_0.01_9645/RNTN_30_tanh_35_5_None_50_0.001_0.01_9645.ckpt
    

.. code:: ipython3

    # Calculate the probabilities for 
    y_probs = model_rntn.predict_proba(np.asarray(X_trees_data).reshape(-1, 1))


.. parsed-literal::

    INFO:tensorflow:Restoring parameters from C:\Users\cskap\github\Springboard\capstone_1\src\models\../../models//RNTN_30_tanh_35_5_None_50_0.001_0.01_9645/RNTN_30_tanh_35_5_None_50_0.001_0.01_9645.ckpt
    

.. code:: ipython3

    # Accuracy 
    print("Accuracy on full test data (RNTN):     {:2f}".format(accuracy_score(y_true, y_pred)))


.. parsed-literal::

    Accuracy on full test data (RNTN):     0.664661
    

*RNTN model accuracy is less than the accuracy of Naive Bayes model.*
Lets look closer at what is mis-classified and also compute other
metrics.

.. code:: ipython3

    # Classification Report
    print(classification_report(y_true, y_pred))


.. parsed-literal::

                  precision    recall  f1-score   support
    
               0       0.32      0.29      0.31      2008
               1       0.38      0.46      0.42      9255
               2       0.88      0.76      0.82     56548
               3       0.30      0.44      0.36     10998
               4       0.48      0.54      0.51      3791
    
       micro avg       0.66      0.66      0.66     82600
       macro avg       0.47      0.50      0.48     82600
    weighted avg       0.71      0.66      0.68     82600
    
    

We now start seeing why this model does better with predicting
sentiments over Naive Bayes. Even though the accuracy is lower, the
per-class model is less confused about classification. It is not
classifying everything is neutral, rather the positive sentiments are
mostly misclassified as slightly positive, which will make prediction
more reliable.

.. code:: ipython3

    # Confusion Matrix
    print(confusion_matrix(y_true, y_pred))


.. parsed-literal::

    [[  591   819   250   288    60]
     [  791  4270  2178  1810   206]
     [  294  4601 43158  8036   459]
     [  116  1183  3355  4852  1492]
     [   36   219   337  1169  2030]]
    

.. code:: ipython3

    # F1-score
    print(f1_score(y_true, y_pred, average='weighted'))


.. parsed-literal::

    0.6836731505540418
    

.. code:: ipython3

    # Log loss per sample
    print(log_loss(y_true, y_probs))


.. parsed-literal::

    1.0353974476756531
    

Again, we see a better F1 score with RNTN model due to better
classification in minority classes. The average Log loss is slightly
higher, due to lesser accuracy of the model.

Root Level Evaluation Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    # Call models predict method
    y_pred = model_rntn.predict(np.asarray(x_test).reshape(-1,1))
    y_true = [t.root.label for t in x_test]


.. parsed-literal::

    INFO:tensorflow:Restoring parameters from C:\Users\cskap\github\Springboard\capstone_1\src\models\../../models//RNTN_30_tanh_35_5_None_50_0.001_0.01_9645/RNTN_30_tanh_35_5_None_50_0.001_0.01_9645.ckpt
    

.. code:: ipython3

    # Calculate probabilities for log loss
    y_probs = model_rntn.predict_proba(np.asarray(x_test).reshape(-1, 1))


.. parsed-literal::

    INFO:tensorflow:Restoring parameters from C:\Users\cskap\github\Springboard\capstone_1\src\models\../../models//RNTN_30_tanh_35_5_None_50_0.001_0.01_9645/RNTN_30_tanh_35_5_None_50_0.001_0.01_9645.ckpt
    

.. code:: ipython3

    # Accuracy 
    print("Accuracy on root test data (RNTN):     {:2f}".format(accuracy_score(y_true, y_pred)))


.. parsed-literal::

    Accuracy on root test data (RNTN):     0.373756
    

Root accuracy for RNTN is significantly higher than baseline. This can
be explained as extreme sentiments are not misclassified as neutral as
much as the nearer class of slightly positive/negative sentiments.

.. code:: ipython3

    # Classification Report
    print(classification_report(y_true, y_pred))


.. parsed-literal::

                  precision    recall  f1-score   support
    
               0       0.29      0.34      0.31       279
               1       0.42      0.48      0.45       633
               2       0.23      0.01      0.01       389
               3       0.29      0.30      0.29       510
               4       0.44      0.68      0.54       399
    
       micro avg       0.37      0.37      0.37      2210
       macro avg       0.33      0.36      0.32      2210
    weighted avg       0.34      0.37      0.33      2210
    
    

.. code:: ipython3

    # Confusion Matrix
    print(confusion_matrix(y_true, y_pred))


.. parsed-literal::

    [[ 96 120   2  40  21]
     [140 301   4 130  58]
     [ 54 141   3 124  67]
     [ 32 119   4 153 202]
     [ 11  33   0  82 273]]
    

.. code:: ipython3

    # F1-score
    print(f1_score(y_true, y_pred, average='weighted'))


.. parsed-literal::

    0.33485051111248126
    

.. code:: ipython3

    # Log loss per sample
    print(log_loss(y_true, y_probs))


.. parsed-literal::

    1.9482420690090614
    

All metrics show better performance as compared to root sentiment
predictions for the baseline model.
