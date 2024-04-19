from lang_processors4.lang_processor import LangProcessor
from lang_processors4.tokenization_utils import (
    process_string,
    replace_tokens,
    indent_lines,
)
import re
from tree_sitter import Language, Parser
from pathlib import Path
from spiral import ronin

NEW_LINE = "NEW_LINE"

COMMENT_TYPES = {"comment", "line_comment", "block_comment"}
IDENTITY_TYPES = {
    "identifier",
    "qualified_identifier",
    "type_identifier",
    "namespace_identifier",
    "field_identifier",
    "statement_identifier"
}

class TreeSitterLangProcessor(LangProcessor):
    def __init__(
            self,
            language,
            ast_nodes_type_string,
            stokens_to_chars,
            chars_to_stokens,
            root_folder,
    ):
        self.language = language
        self.ast_nodes_type_string = ast_nodes_type_string
        self.stokens_to_chars = stokens_to_chars
        self.chars_to_stokens = chars_to_stokens
        self.root_folder = Path(root_folder)
        self.root_folder.is_dir(), f"{self.root_folder} is not a directory."
        self.parser = None
        self.create_treesiter_parser()

    def create_treesiter_parser(self):
        if self.parser is None:
            lib_path = self.root_folder.joinpath(f"{self.language}.so")
            repo_path = self.root_folder.joinpath(f"tree-sitter-{self.language}")
            if not lib_path.exists():
                assert repo_path.is_dir()
                Language.build_library(
                    # Store the library in the `build` directory
                    str(lib_path),
                    # Include one or more languages
                    [str(repo_path)],
                )
            language = Language(str(lib_path), self.language)
            self.parser = Parser()
            self.parser.set_language(language)

    def tokenize_code(self, code, keep_comments=False, process_strings=True):
        tokenized_code = []
        tokens, token_types, position = self.get_tokens_and_types(code)
        last_lineno = -1
        last_col = 0
        for token, token_type, position in zip(tokens, token_types, position):
            # print(token, token_type)
            start_line, start_col = position[0]
            end_line, end_col = position[1]
            if start_line > last_lineno:
                tokenized_code.append('\n')
                last_col = 0
            if start_col > last_col:
                tokenized_code.append((" " * (start_col - last_col)))
            if token_type in COMMENT_TYPES and keep_comments is False:
                continue
            if token_type in self.ast_nodes_type_string:
                token = process_string(
                    token,
                    self.chars_to_stokens,
                    self.stokens_to_chars,
                    token_type in COMMENT_TYPES,
                    process_strings,
                )
            if token_type in IDENTITY_TYPES:
                splited_tokens = ronin.split(token)
                token = ' '.join(splited_tokens)
            # if token_type == 'number_literal':
            #     token = '<num>'
            if len(token) > 0:
                tokenized_code.append(token)
            last_col = end_col
            last_lineno = end_line
        
        line = []
        lines = []
        for token in tokenized_code:
            if token == '\n':
                code = ' '.join(line)
                lines.append(code)
                line = []
            else:
                if token != '':
                    line.append(token)
        if line != []:
            code = ' '.join(line)
            lines.append(code)
        line_list = [line for line in lines if line.strip() != '']
        out = '\n'.join(line_list)
        return out

    def get_tokens_and_types(self, code):
        code = code.replace("\r", "")
        # print(code)
        code = bytes(code, "utf8", errors='ignore')
        tree = self.get_ast(code)
        tokens = []
        tokens_type = []
        position = []
        self.dfs(code, tree.root_node, tokens, tokens_type, position)
        return tokens, tokens_type, position

    def get_ast(self, code):
        assert isinstance(code, str) or isinstance(code, bytes)
        if isinstance(code, str):
            code = bytes(code, "utf8", errors='ignore')
        tree = self.parser.parse(code)
        return tree

    def dfs(self, code, node, tokens, tokens_type, position):
        # print(node)
        if len(node.children) == 0 or node.type in self.ast_nodes_type_string:
            snippet = code[node.start_byte: node.end_byte]
            if isinstance(snippet, bytes):
                snippet = snippet.decode("utf8", errors='ignore')
            if len(snippet) > 0:
                tokens.append(snippet)
                tokens_type.append(node.type)
                position.append([node.start_point, node.end_point])
            # tokens.append(snippet)
            # tokens_type.append(node.type)
            # position.append([node.start_point, node.end_point])
            return
        for child in node.children:
            self.dfs(code, child, tokens, tokens_type, position)

    def detokenize_code(self, code):
        # TODO make this cleaner with tree sitter AST ?
        assert isinstance(code, str) or isinstance(code, list)
        if isinstance(code, list):
            code = " ".join(code)
        code = code.replace("ENDCOM", "\n")
        replaced_tokens = []
        # call parser of the tokenizer to find comments and string and detokenize them correctly
        try:
            tokens, token_types = self.get_tokens_and_types(code)
            # print(tokens)
            # print(token_types)
            for token, token_type in zip(tokens, token_types):
                if token_type in self.ast_nodes_type_string:
                    token_ = token.replace("STRNEWLINE", "\n").replace(
                        "TABSYMBOL", "\t"
                    )
                    token_ = (
                        replace_tokens(token_, self.chars_to_stokens)
                            .replace(" ", "")
                            .replace("▁", " ")
                    )
                    if token_type in COMMENT_TYPES:
                        token_ += "\n"
                    replaced_tokens.append(token_)
                else:
                    replaced_tokens.append(token)
        except KeyboardInterrupt:
            raise
        except:
            pass

        code = " ".join(replaced_tokens)
        code = code.replace("\n", "NEW_LINE")
        code = code.replace('} "', 'CB_ "')
        code = code.replace('" {', '" OB_')
        code = code.replace("*/ ", "*/ NEW_LINE")
        code = code.replace("} ;", "CB_COLON NEW_LINE")
        code = code.replace("} ,", "CB_COMA")
        code = code.replace("}", "CB_ NEW_LINE")
        code = code.replace("{", "OB_ NEW_LINE")
        code = code.replace(";", "; NEW_LINE")
        code = replace_tokens(code, self.stokens_to_chars)
        lines = re.split("NEW_LINE", code)

        untok_s = indent_lines(lines)
        untok_s = (
            untok_s.replace("CB_COLON", "};")
                .replace("CB_COMA", "},")
                .replace("CB_", "}")
                .replace("OB_", "{")
        )
        untok_s = untok_s.replace("> > >", ">>>").replace("<< <", "<<<")
        untok_s = untok_s.replace("> >", ">>").replace("< <", "<<")

        return untok_s

    def extract_arguments_using_parentheses(self, function):
        function = function.split(" ")
        types = []
        names = []
        par = 0
        arguments = []
        function = function[function.index("("):]
        for tok in function:
            if tok == "(":
                par += 1
            elif tok == ")":
                par -= 1
            arguments.append(tok)
            if par == 0:
                break
        arguments = " ".join(arguments[1:-1])
        if arguments == "":
            return ["None"], ["None"]
        arguments = arguments.split(",")
        for arg in arguments:
            bracks = re.findall("\[ \]", arg)
            bracks = " ".join(bracks)
            arg = arg.replace(bracks, "")
            arg = arg.strip()
            arg = re.sub(" +", " ", arg)
            t = " ".join(arg.split(" ")[:-1] + [bracks])
            n = arg.split(" ")[-1]
            types.append(t)
            names.append(n)
        return types, names

    def get_first_token_before_first_parenthesis(self, code):
        assert isinstance(code, str) or isinstance(
            code, list
        ), f"function is not the right type, should be str or list : {code}"
        if isinstance(code, str):
            code = code.split()
        return code[code.index("(") - 1]
