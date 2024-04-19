from lang_processors2.tree_sitter_processor import (
    TreeSitterLangProcessor,
)
from lang_processors.java_processor import (
    JAVA_TOKEN2CHAR,
    JAVA_CHAR2TOKEN,
)

IDENTIFIERS = {"identifier", "field_identifier"}

C_TOKEN2CHAR = JAVA_TOKEN2CHAR.copy()
C_CHAR2TOKEN = JAVA_CHAR2TOKEN.copy()


class CProcessor(TreeSitterLangProcessor):
    def __init__(self, root_folder):
        super().__init__(
            language="c",
            ast_nodes_type_string=["comment", "string_literal", "char_literal", "system_lib_string"],
            stokens_to_chars=C_TOKEN2CHAR,
            chars_to_stokens=C_CHAR2TOKEN,
            root_folder=root_folder,
        )