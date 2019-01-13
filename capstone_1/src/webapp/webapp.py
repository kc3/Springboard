# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.getcwd())

from src.models.predict_model import predict_model
from flask import Flask, render_template, request


def create_app(config=None):
    app = Flask(__name__)

    # See http://flask.pocoo.org/docs/latest/config/
    app.config.update(dict(DEBUG=True))
    app.config.update(config or {})

    # Definition of the routes. Put them into their own file. See also
    # Flask Blueprints: http://flask.pocoo.org/docs/latest/blueprints
    @app.route("/", methods=['GET', 'POST'])
    def sentiment():
        if request.method == 'POST':
            review_text = request.form['text']
            label, tree_txt = predict_model(review_text)
            return render_template('sentiment.html', text=review_text, label=label, tree_txt=tree_txt)
        else:
            return render_template('sentiment.html', text=None, label=None, tree_txt=None)

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app = create_app()
    app.run(host="127.0.0.1", port=port)
