from io import StringIO
import tokenize
from lang_processors2.tokenization_utils import (
    process_string,
)
def py_tokenize(source, process_strings=True):
    """
    Returns 'source' minus comments and docstrings.
    """
    io_obj = StringIO(source)
    token_list = []
    prev_tokentype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    spetoken2char = {
            "STOKEN00": "#",
            "STOKEN1": "\\n",
            "STOKEN2": '"""',
            "STOKEN3": "'''",
    }
    char2spetoken = {
        value: " " + key + " " for key, value in spetoken2char.items()
    }
    
    try:
        for token in tokenize.generate_tokens(io_obj.readline):
            token_type = token[0]
            token_string = token[1]
            start_line, start_col = token[2]
            end_line, end_col = token[3]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                token_list.append((" " * (start_col - last_col)))
            if token_type == tokenize.COMMENT:
                pass
            elif token_type == tokenize.STRING:
                if prev_tokentype != tokenize.INDENT and prev_tokentype != tokenize.NEWLINE and start_col > 0:
                    token_list.append(
                        process_string(
                            token_string,
                            char2spetoken,
                            spetoken2char,
                            False,
                            do_whole_processing=process_strings,
                        )
                )
            else:
                token_list.append(token_string)
            prev_tokentype = token_type
            last_col = end_col
            last_lineno = end_line
    except (
            IndentationError,
            SyntaxError,
            UnicodeDecodeError,
    ) as e:
        print("Tokenization error occurred:", e)
        return False
    except tokenize.TokenError as e:
        print("Tokenization error occurred:", e)
        pass
    line = []
    lines = []
    for token in token_list:
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
    