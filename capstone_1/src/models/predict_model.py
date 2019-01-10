# -*- coding: utf-8 -*-

#
# Functionality to predict by using a trained model
#

from src.models.rntn import RNTN


def predict_model(x, model_name='RNTN_100_relu_10_1_0.1_0.1'):
    """ Predict model based on input value.

    :param x:
        A single review text string. Can be multiple sentences.
    :param model_name:
        Trained model name (should be present in models folder)
    :return:
        Sentiment label for the text.
    """
    r = RNTN(model_name=model_name)
    y_pred = r.predict(x)
    return y_pred
