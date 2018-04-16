import pickle
from collections import defaultdict

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from gensim.models.wrappers.fasttext import FastText

from file_path_manager import FilePathManager


class Corpus:
    START_SYMBOL = "<start>"
    END_SYMBOL = "<end>"
    UNK = "<unk>"
    PAD = "<pad>"

    def __init__(self, word2idx=None, idx2word=None, word_embeddings=None):
        self.word2idx = word2idx if word2idx is not None else {}
        self.idx2word = idx2word if idx2word is not None else {}
        self.fast_text = word_embeddings if word_embeddings is not None else {}
        self.vocab_size = len(self.word2idx)
        self.embed_size = 300
        self.max_sentence_length = 16
        self.min_word_freq = 5

    def word_embedding(self, word):
        if word not in self.word2idx and word not in self.idx2word:
            word = self.UNK
        if isinstance(word, int):
            word = self.word_from_index(word)
        result = torch.from_numpy(self.fast_text[word]).view(-1)
        return result

    def word_one_hot(self, word):
        if word not in self.word2idx and word not in self.idx2word:
            word = self.UNK
        result = torch.zeros(self.vocab_size).view(-1)
        if isinstance(word, str):
            word = self.word_index(word)
        result[word] = 1
        return result

    def word_index(self, word):
        return self.word2idx[word]

    def word_from_index(self, index):
        return self.idx2word[index]

    def add_word(self, word: str):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.word2idx)
            self.idx2word[len(self.idx2word)] = word
            self.vocab_size += 1

    def prepare(self):
        self.word2idx = defaultdict(int)
        # to make sure start_symbol, end_symbol, pad, and unk will be included
        self.word2idx[self.START_SYMBOL] = self.word2idx[self.END_SYMBOL] = self.word2idx[self.UNK] = self.word2idx[
            self.PAD] = self.min_word_freq
        for dataset_type in ["train", "val"]:
            caps = dset.CocoCaptions(root=FilePathManager.resolve(f'data/{dataset_type}'),
                                     annFile=FilePathManager.resolve(
                                         f"data/annotations/captions_{dataset_type}2017.json"),
                                     transform=transforms.ToTensor())
            for _, captions in caps:
                for capt in captions:
                    tokens = self.tokenize(capt)
                    for token in tokens:
                        self.word2idx[token] += 1
        temp = {}
        embeddings = {}
        fast_text = FastText.load(FilePathManager.resolve("data/fasttext.model"), mmap="r")
        for k, v in self.word2idx.items():
            if v >= self.min_word_freq:
                temp[k] = len(temp)
                embeddings[k] = fast_text[k] if k in fast_text else fast_text[self.UNK]
        self.word2idx = temp
        # swap keys and values
        self.idx2word = dict(zip(self.word2idx.values(), self.word2idx.keys()))
        self.fast_text = embeddings

    @staticmethod
    def remove_nonalpha(word: str):
        return word.strip().strip(".")

    @staticmethod
    def preprocess_sentence(sentence: str):
        sentence = sentence.lower().strip().strip(".").replace("'", "").replace(",", " , ").replace("\"", "")
        return sentence

    def tokenize(self, sentence: str):
        temp = self.preprocess_sentence(sentence).split(" ")
        return [self.remove_nonalpha(x)
                for x in temp
                if not x.isspace()]

    def pad_sentence(self, tokens):
        tokens = tokens[:self.max_sentence_length]
        temp = len(tokens)
        if temp != self.max_sentence_length:
            tokens.extend([self.PAD] * (self.max_sentence_length - temp))
        return tokens

    def embed_sentence(self, sentence: str, one_hot=False, pad: bool = True):
        tokens = self.tokenize(sentence)
        if pad:
            tokens = self.pad_sentence(tokens)
        result = torch.zeros(self.max_sentence_length, self.vocab_size if one_hot else self.embed_size)
        for i in range(self.max_sentence_length):
            result[i] = self.word_one_hot(tokens[i]) if one_hot else self.word_embedding(tokens[i])
        return result

    def __call__(self, sentence, one_hot: bool = False):
        return self.embed_sentence(sentence, one_hot)

    def store(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump((self.word2idx, self.idx2word, self.fast_text), f)

    @staticmethod
    def load(file_path):
        with open(file_path, "rb") as f:
            word2idx, idx2word, fast_text = pickle.load(f)
        return Corpus(word2idx, idx2word, fast_text)


if __name__ == '__main__':
    corpus = Corpus()
    corpus.prepare()
    corpus.store(FilePathManager.resolve("data/corpus.pkl"))
    # corpus = Corpus.load(FilePathManager.resolve("data/corpus.pkl"))
    # print(corpus.word_one_hot("test"))
    # print(corpus.vocab_size)
