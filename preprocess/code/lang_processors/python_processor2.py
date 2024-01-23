from lang_processors.tokenization_utils import (
    process_string,
)

from lang_processors.lang_processor import LangProcessor

import tokenize
from io import BytesIO, StringIO
import subprocess

class PythonProcessor(LangProcessor):
    def __init__(self, root_folder=None):
        self.spetoken2char = {
            "STOKEN00": "#",
            "STOKEN1": "\\n",
            "STOKEN2": '"""',
            "STOKEN3": "'''",
        }
        self.char2spetoken = {
            value: " " + key + " " for key, value in self.spetoken2char.items()
        }
        self.language = "python"
        
    def formatter(self, file_path):
        with open(file_path, 'r+', encoding='utf-8') as f:
            source = f.read()
            io_obj = StringIO(source)
            prev_tokentype = tokenize.INDENT
            removed_docstr = 0

            token_list = []
            for token in tokenize.generate_tokens(io_obj.readline):
                token_type = token[0]
                token_string = token[1]
                start_line, start_col = token[2]
                if token_type == tokenize.ENCODING or token_type == tokenize.COMMENT or token_type == tokenize.NL:
                    pass
                elif token_type == tokenize.NEWLINE:
                    if removed_docstr == 1:
                        removed_docstr = 0
                    else:
                        token_list.append("NEW_LINE")
                elif token_type == tokenize.STRING:
                    if prev_tokentype != tokenize.INDENT and prev_tokentype != tokenize.NEWLINE and start_col > 0:
                        token_list.append(token_string)
                    else:
                        removed_docstr = 1
                elif token_type == tokenize.INDENT:
                    token_list.append("INDENT")
                elif token_type == tokenize.DEDENT:
                    # empty block
                    if token_list[-1] == "INDENT":
                        token_list = token_list[:-1]
                    else:
                        token_list.append("DEDENT")
                else:
                    token_list.append(token_string)
                prev_tokentype = token_type
                code = " ".join(token_list)
                
            lines = code.split("NEW_LINE")
            tabs = ""
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith("INDENT "):
                    tabs += "    "
                    line = line.replace("INDENT ", tabs)
                elif line.startswith("DEDENT"):
                    number_dedent = line.count("DEDENT")
                    tabs = tabs[4 * number_dedent:]
                    line = line.replace("DEDENT", "")
                    line = line.strip()
                    line = tabs + line
                elif line == "DEDENT":
                    line = ""
                else:
                    line = tabs + line
                lines[i] = line
            formatted_source = "\n".join(lines)
            f.seek(0)
            f.truncate()
            f.seek(0)
            f.write(formatted_source)
        return formatted_source
    
    def diff(self, file_path='before.txt', file_path2='after.txt'):
        added_code_num = []
        deleted_code_num = []
        command = [
            'diff',
            '--old-line-format=-\t%dn\t%L',
            '--new-line-format=+\t%dn\t%L',
            '--unchanged-line-format=',
            '-w',
            '-B',
            file_path,
            file_path2
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        output = process.communicate()[0]
        result = output.decode('utf-8','ignore').strip().split('\n')
        for line in result:
            tmp = line.split('\t')
            if tmp[2] != '':
                if tmp[0] == '+':
                    added_code_num.append(tmp[1])
                else:
                    deleted_code_num.append(tmp[1])
        
        return added_code_num, deleted_code_num
        
    def tokenize_code(self, code, process_strings=True):
        assert isinstance(code, str)
        code = code.replace(r"\r", "")
        code = code.replace("\r", "")
        tokens = []

        try:
            iterator = tokenize.tokenize(BytesIO(code.encode("utf-8")).readline)
        except SyntaxError as excep:
            raise SyntaxError(excep)

        removed_docstr = 0
        while True:
            try:
                toktype, tok, _, _, line = next(iterator)
            except (
                    tokenize.TokenError,
                    IndentationError,
                    SyntaxError,
                    UnicodeDecodeError,
            ) as e:
                raise ValueError(
                    f'Impossible to parse tokens because of incorrect source code "{e}" ...'
                )
            except StopIteration:
                raise Exception(f"End of iterator before ENDMARKER token.")

            if toktype == tokenize.ENCODING or toktype == tokenize.NL:
                continue

            elif toktype == tokenize.NEWLINE:
                if removed_docstr == 1:
                    removed_docstr = 0
                    continue
                tokens.append("NEW_LINE")

            elif toktype == tokenize.STRING:
                tokens.append(
                    process_string(
                        tok,
                        self.char2spetoken,
                        self.spetoken2char,
                        False,
                        do_whole_processing=process_strings,
                    )
                )

            elif toktype == tokenize.INDENT:
                tokens.append("INDENT")

            elif toktype == tokenize.DEDENT:
                # empty block
                if tokens[-1] == "INDENT":
                    tokens = tokens[:-1]
                else:
                    tokens.append("DEDENT")

            elif toktype == tokenize.ENDMARKER:
                tokens.append("ENDMARKER")
                break

            else:
                tokens.append(tok)

        assert tokens[-1] == "ENDMARKER", "Error, no end marker"
        return tokens[:-1]
    
    def detokenize_code(self, code):
        # replace recreate lines with \n and appropriate indent / dedent
        # removing indent/ dedent tokens
        assert isinstance(code, str) or isinstance(code, list)
        if isinstance(code, list):
            code = " ".join(code)
        code = code.replace("ENDCOM", "NEW_LINE")
        code = code.replace("▁", "SPACETOKEN")
        lines = code.split("NEW_LINE")
        tabs = ""
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("INDENT "):
                tabs += "    "
                line = line.replace("INDENT ", tabs)
            elif line.startswith("DEDENT"):
                number_dedent = line.count("DEDENT")
                tabs = tabs[4 * number_dedent:]
                line = line.replace("DEDENT", "")
                line = line.strip()
                line = tabs + line
            elif line == "DEDENT":
                line = ""
            else:
                line = tabs + line
            lines[i] = line
        untok_s = "\n".join(lines)
        # find string and comment with parser and detokenize string correctly
        try:
            for toktype, tok, _, _, line in tokenize.tokenize(
                    BytesIO(untok_s.encode("utf-8")).readline
            ):
                if toktype == tokenize.STRING or toktype == tokenize.COMMENT:
                    tok_ = (
                        tok.replace("STRNEWLINE", "\n")
                            .replace("TABSYMBOL", "\t")
                            .replace(" ", "")
                            .replace("SPACETOKEN", " ")
                    )
                    untok_s = untok_s.replace(tok, tok_)
        except KeyboardInterrupt:
            raise
        except:
            # TODO raise ValueError(f'Invalid python function \n {code}\n')
            pass
        # detokenize imports
        untok_s = (
            untok_s.replace(". ", ".")
                .replace(" .", ".")
                .replace("import.", "import .")
                .replace("from.", "from .")
        )
        # special strings
        string_modifiers = ["r", "u", "f", "rf", "fr", "b", "rb", "br"]
        for modifier in string_modifiers + [s.upper() for s in string_modifiers]:
            untok_s = untok_s.replace(f" {modifier} '", f" {modifier}'").replace(
                f' {modifier} "', f' {modifier}"'
            )
        untok_s = untok_s.replace("> >", ">>").replace("< <", "<<")
        return untok_s
    
