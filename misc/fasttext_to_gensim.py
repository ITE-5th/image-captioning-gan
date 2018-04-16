import gensim

from gensim.models.wrappers.fasttext import FastText

from file_path_manager import FilePathManager

if __name__ == '__main__':
    model = FastText.load_fasttext_format(FilePathManager.resolve("data/wiki.en"))
    model.save(FilePathManager.resolve("data/fasttext.model"))