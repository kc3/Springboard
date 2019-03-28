Getting started
===============

Download Source
~~~~~~~~~~~~~~~~

> *git clone https://github.com/kc3/Springboard.git*

Change directory
~~~~~~~~~~~~~~~~

> *cd capstone_2*


Create Environment
~~~~~~~~~~~~~~~~~~~

> *conda env create -n 'capstone_2' -f environment.yml*


Test Environment
~~~~~~~~~~~~~~~~~~~

> *tox*

Download Movie Dialog Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

> *python .\src\data\make_dataset.py ./src/data/raw ./src/data/interim ./src/data/processed*

Chat Demo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

> *python -m src.models.predict*
