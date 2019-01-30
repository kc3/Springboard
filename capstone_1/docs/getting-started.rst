Getting started
===============

Download Source
~~~~~~~~~~~~~~~~

> *git clone https://github.com/kc3/Springboard.git*

Change directory
~~~~~~~~~~~~~~~~

> *cd capstone_1*


Create Environment
~~~~~~~~~~~~~~~~~~~

> *conda env create -n 'capstone_1' -f environment-cpu.yml*


Test Environment
~~~~~~~~~~~~~~~~~~~

> *tox*

Download CoreNLP
~~~~~~~~~~~~~~~~~~~

> Download and unzip corenlp from *http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip*

Start CoreNLP
~~~~~~~~~~~~~~~~~~~

> *java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000*

Start Prediction server
~~~~~~~~~~~~~~~~~~~~~~~

> *python -m src.webapp.webapp*

View Results
~~~~~~~~~~~~~~~~~~~

> Enter your movie review in the text box to view results!




