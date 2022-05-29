#!/usr/bin/env bash

sudo apt update
pip install --upgrade pip
# required packages
pip install pandas numpy scipy spacy tqdm spacytextblob vaderSentiment networkx scikit-learn tensorflow gensim textsearch contractions nltk beautifulsoup4 transformers autocorrect pytesseract opencv-python
# install spacy model
python -m spacy download en_core_web_sm
# ocr tools
sudo apt install -y tesseract-ocr
sudo apt install -y libtesseract-dev