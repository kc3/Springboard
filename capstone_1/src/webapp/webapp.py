# -*- coding: utf-8 -*-

import os

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
            return render_template('sentiment.html', text=request.form['text'])
        else:
            return render_template('sentiment.html', text=None)

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app = create_app()
    app.run(host="127.0.0.1", port=port)
