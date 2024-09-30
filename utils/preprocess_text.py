import nltk
import numpy as np
import re
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from utils.constants.patterns import RE_PATTERNS, STOPWORD_LIST, APPO
import os

print(os.getcwd())

# Ensure you have downloaded necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')


class TextPreprocessor:
    def __init__(self, tokenizer_path, max_pad_len):
        self.max_pad_len = max_pad_len
        self.tokenizer = self.load_tokenizer(tokenizer_path)
        self.lemmatizer = WordNetLemmatizer()

    def load_tokenizer(self, tokenizer_path):
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return tokenizer

    def clean_text(self, text, remove_repeat_text=True, remove_patterns_text=True):
        text = text.lower()
        words = text.split(" ")
        text = [APPO[word] if word in APPO else word for word in words]
        text = " ".join(text)

        if remove_patterns_text:
            for target, patterns in RE_PATTERNS.items():
                for pat in patterns:
                    text = str(text).replace(pat, target)

        if remove_repeat_text:
            text = re.sub(r'(.)\1{2,}', r'\1', text)  # Replace repeating characters

        text = str(text).replace("\n", " ")
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub('[0-9]', "", text)
        text = re.sub(" +", " ", text)
        text = re.sub("([^\x00-\x7F])+"," ", text)
        return text

    def lemmatize_text(self, text, lemmatization=True):
        output = ''
        if lemmatization:
            text = text.split(' ')
            for word in text:
                word1 = self.lemmatizer.lemmatize(word, pos="n")  # noun
                word2 = self.lemmatizer.lemmatize(word1, pos="v")  # verb
                word3 = self.lemmatizer.lemmatize(word2, pos="a")  # adjective
                word4 = self.lemmatizer.lemmatize(word3, pos="r")  # adverb
                output += " " + word4
        else:
            output = text
        return str(output.strip())

    def remove_stopwords(self, text):
        output = ""
        words = text.split(" ")
        for word in words:
            if word not in STOPWORD_LIST:
                output += " " + word
        return output.strip()

    def tokenize_and_pad(self, text):
        list_tokenized = self.tokenizer.texts_to_sequences([text])
        training_padded = pad_sequences(list_tokenized, maxlen=self.max_pad_len, padding='post')
        return training_padded

    def preprocess(self, text):
        cleaned_text = self.clean_text(text)
        lemmatized_text = self.lemmatize_text(cleaned_text)
        text_without_stopwords = self.remove_stopwords(lemmatized_text)
        tokenized_padded_text = self.tokenize_and_pad(text_without_stopwords)
        return tokenized_padded_text
