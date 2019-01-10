# -*- coding: utf-8 -*-

import pytest
from src.webapp import webapp


@pytest.fixture
def app():
    app = webapp.create_app()
    app.debug = True
    return app.test_client()


def test_sentiment(app):
    res = app.get("/")
    # print(dir(res), res.status_code)
    assert res.status_code == 200
    assert b"Hello World" in res.data
