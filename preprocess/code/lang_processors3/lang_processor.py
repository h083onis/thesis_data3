from abc import ABC


class LangProcessor(ABC):
    processors = {}

    @classmethod
    def __init_subclass__(cls):
        super().__init_subclass__()
        assert (
                len(cls.__name__.lower().split("processor")) == 2
                and cls.__name__.lower().split("processor")[1] == ""
        ), "language processors class name should be that format : YourlanguageProcessor"
        cls.processors[cls.__name__.lower().split("processor")[0]] = cls

    def tokenize_code(self, code, keep_comments=False, process_strings=True):
        raise NotImplementedError

    def detokenize_code(self, code):
        raise NotImplementedError

    def obfuscate_code(self, code):
        raise NotImplementedError

    def extract_functions(self, code):
        raise NotImplementedError

    def extract_function_name(self, function):
        raise NotImplementedError

    def extract_arguments(self, function):
        raise NotImplementedError
