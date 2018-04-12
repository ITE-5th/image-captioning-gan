import os


class FilePathManager:
    base_dir = os.path.dirname(os.path.abspath(__file__))

    @staticmethod
    def resolve(path):
        return f"{FilePathManager.base_dir}/{path}"
