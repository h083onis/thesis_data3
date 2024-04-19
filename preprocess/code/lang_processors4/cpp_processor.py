from lang_processors4.tree_sitter_processor import (
    TreeSitterLangProcessor,
)
from lang_processors2.java_processor import (
    JAVA_TOKEN2CHAR,
    JAVA_CHAR2TOKEN,
)

IDENTIFIERS = {"identifier", "field_identifier"}

CPP_TOKEN2CHAR = JAVA_TOKEN2CHAR.copy()
CPP_CHAR2TOKEN = JAVA_CHAR2TOKEN.copy()


class CppProcessor(TreeSitterLangProcessor):
    def __init__(self, root_folder):
        super().__init__(
            language="cpp",
            ast_nodes_type_string=["comment", "string_literal", "char_literal", "system_lib_string"],
            stokens_to_chars=CPP_TOKEN2CHAR,
            chars_to_stokens=CPP_CHAR2TOKEN,
            root_folder=root_folder,
        )