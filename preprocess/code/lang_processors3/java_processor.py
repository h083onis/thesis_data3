from lang_processors3.tree_sitter_processor import (
    TreeSitterLangProcessor,
)

JAVA_TOKEN2CHAR = {
    "STOKEN00": "//",
    "STOKEN01": "/*",
    "STOKEN02": "*/",
    "STOKEN03": "/**",
    "STOKEN04": "**/",
    "STOKEN05": '"""',
    "STOKEN06": "\\n",
    "STOKEN07": "\\r",
    "STOKEN08": ";",
    "STOKEN09": "{",
    "STOKEN10": "}",
    "STOKEN11": r"\'",
    "STOKEN12": r"\"",
    "STOKEN13": r"\\",
}
JAVA_CHAR2TOKEN = {value: " " + key + " " for key, value in JAVA_TOKEN2CHAR.items()}


class JavaProcessor(TreeSitterLangProcessor):
    def __init__(self, root_folder):
        super().__init__(
            language="java",
            ast_nodes_type_string=["comment", "string_literal", "character_literal"],
            stokens_to_chars=JAVA_TOKEN2CHAR,
            chars_to_stokens=JAVA_CHAR2TOKEN,
            root_folder=root_folder,
        )