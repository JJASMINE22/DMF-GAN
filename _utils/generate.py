import os
import re
import json
import random
import numpy as np
from PIL import Image
from configs import config as cfg
from _utils.utils import image_preprocess

class DataGenerator:
    def __init__(self,
                 root_path: str,
                 token_size: int,
                 batch_size: int,
                 train_ratio: float):
        self.root_path = root_path
        self.token_size = token_size
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.split_train_val()
        self.tokens = self.get_tokens()

    def split_train_val(self):

        file_paths, text_paths = list(), list()
        for root, _, files in os.walk(self.root_path):

            for file in files:
                if file[-3:] == 'jpg':
                    file_paths.append(os.path.join(root, file))
                else:
                    text_paths.append(os.path.join(root, file))

        self.train_file_paths = file_paths[: int(self.train_ratio*file_paths.__len__())]
        self.train_text_paths = text_paths[: int(self.train_ratio*text_paths.__len__())]
        self.validate_file_paths = file_paths[int(self.train_ratio*file_paths.__len__()):]
        self.validate_text_paths = text_paths[int(self.train_ratio*text_paths.__len__()):]

    def get_train_len(self):

        if not self.train_file_paths.__len__() % self.batch_size:
            return self.train_file_paths.__len__() // self.batch_size
        else:
            return self.train_file_paths.__len__() // self.batch_size + 1

    def get_val_len(self):

        if not self.validate_file_paths.__len__() % self.batch_size:
            return self.validate_file_paths.__len__() // self.batch_size
        else:
            return self.validate_file_paths.__len__() // self.batch_size + 1

    @classmethod
    def word_preprocess(cls, sentence):
        """
        Completely exclude special characters
        Adding characters like 's、're etc.
        Phrase terminators are taken into account
        """
        words = []
        for word in sentence.split():
            tokens = re.sub(r"[^a-zA-Z0-9 ]", r" ", word).split()
            marks = [mark.start() for mark in re.finditer("'", word)]

            for i, mark in enumerate(marks):
                try:
                    tokens[i + 1] = word[mark] + tokens[i + 1]
                except IndexError:
                    continue

            if word[-1] in ['.', ',', '?', '!']:
                tokens.append(word[-1])
            words.extend(tokens)

        return words

    def get_tokens(self):

        try:
            with open(cfg.token_path, 'r') as f:
                dict = json.load(f)
        except FileNotFoundError:
            pass
        else:
            return dict['tokens']

        text_paths = self.train_text_paths + self.validate_text_paths

        tokens = list()
        for text_path in text_paths:
            f = open(text_path, 'r')
            for sentence in f.readlines():
                tokens.extend(DataGenerator.word_preprocess(sentence.strip()))
        unique_tokens = list(set(tokens))

        counts = list()
        for token in unique_tokens:
            counts.append(tokens.count(token))

        index = sorted(np.arange(len(unique_tokens)), key=lambda i: counts[i], reverse=True)

        sorted_tokens = ['', 'UNK'] + list(np.array(unique_tokens)[index])

        tokens = sorted_tokens[:self.token_size]

        dict = {'tokens': tokens}
        with open(cfg.token_path, 'w') as f:
            json.dump(dict, f, indent=4)

        return tokens

    def data_generate(self, training: bool=True):
        """
        note: The len of the seq of batch is determined by the longest text in it
        """

        if training:
            file_paths = self.train_file_paths
            text_paths = self.train_text_paths
            random_index = np.random.choice(file_paths.__len__(),
                                            file_paths.__len__(),
                                            replace=False)
            file_paths = np.array(file_paths)[random_index]
            text_paths = np.array(text_paths)[random_index]

        else:
            file_paths = self.validate_file_paths
            text_paths = self.validate_text_paths
            random_index = np.random.choice(file_paths.__len__(),
                                            file_paths.__len__(),
                                            replace=False)
            file_paths = np.array(file_paths)[random_index]
            text_paths = np.array(text_paths)[random_index]

        while True:
            text_sources, former_sources, latter_sources = [], [], []
            for index, (file_path, text_path) in enumerate(zip(file_paths, text_paths)):

                text_source = []
                sentences = open(text_path, 'r').readlines()
                sentence = random.choice(sentences)
                for token in DataGenerator.word_preprocess(sentence.strip()):
                    try:
                        text_source.append(self.tokens.index(token))
                    except ValueError:
                        text_source.append(self.tokens.index('UNK'))

                image = Image.open(file_path).convert(mode='RGB')

                former_image = image.resize(cfg.former_size)
                latter_image = image.resize(cfg.latter_size)
                former_source = image_preprocess(former_image)
                latter_source = image_preprocess(latter_image)

                text_sources.append(text_source.copy())
                former_sources.append(former_source)
                latter_sources.append(latter_source)

                if np.logical_or(np.equal(text_sources.__len__(), self.batch_size),
                                 np.equal(index, file_paths.__len__() - 1)):

                    max_len = max(map(lambda source: source.__len__(), text_sources))
                    [source.extend([self.tokens.index('')] * (max_len - len(source))) for source in text_sources]

                    annotation_text_srcs, annotation_former_srcs, annotation_latter_srcs = text_sources.copy(), \
                                                                                           former_sources.copy(),\
                                                                                           latter_sources.copy()
                    text_sources.clear()
                    former_sources.clear()
                    latter_sources.clear()

                    yield np.array(annotation_text_srcs), np.array(annotation_former_srcs), np.array(annotation_latter_srcs)
