def c_formatter(tokens, tokens_types):
    lines = []
    line = []
    include_flag = 0
    for_flag = 0
    if_flag = 0
    while_flag = 0
    parentheses_flag = 0
    else_flag = 0
    do_flag = 0
    case_default_flag = 0
    for token, token_type in zip(tokens, tokens_types):
        if (token_type != '{' and token_type != ';') and (if_flag == 2 or for_flag == 2 or while_flag == 2):
            lines.append(line)
            line = []
            if_flag = 0
            for_flag = 0
            while_flag = 0
        
        # else
        if (token_type != 'if' and token_type !='{') and (else_flag == 1 or do_flag == 1):
            lines.append(line)
            line = []
            else_flag = 0
            do_flag = 0
        
        if token_type == '{':
            line.append(token)
            lines.append(line)
            line = []
            for_flag = 0
            if_flag = 0
            else_flag = 0
            do_flag = 0
            while_flag = 0
        elif token_type == '}':
            lines.append(line)
            lines.append(token)
            line = []
        elif token_type == 'if':
            line.append(token)
            if_flag = 1
            else_flag = 0
        elif token_type == 'else':
            line.append(token)
            else_flag = 1
        elif token_type == 'do':
            line.append(token)
            do_flag = 1
        elif token_type == 'while':
            line.append(token)
            while_flag = 1
        elif token_type == 'case' or token_type == 'defalut':
            line.append(token)
            case_default_flag = 1
        elif token_type == ':':
            line.append(token)
            lines.append(line)
            line = []
            case_default_flag = 0
        elif token_type == '(' and (if_flag == 1 or for_flag == 1 or while_flag == 1):
            line.append(token)
            parentheses_flag += 1
        elif token_type == ')' and (if_flag == 1 or for_flag == 1 or while_flag == 1):
            line.append(token)
            parentheses_flag -= 1
            if parentheses_flag == 0:
                if_flag = 2
                for_flag == 2
                while_flag == 2
        elif token_type == '#include':
            line.append(token)
            include_flag = 1
        elif token_type == 'string_literal' or token_type == 'system_lib_string':
            line.append(token)
            if include_flag == 1:
                lines.append(line)
                line = []
                include_flag = 0
        elif token_type == 'for':
            line.append(token)
            for_flag = 1
        elif token_type == ';':
            line.append(token)
            if for_flag == 0:
                lines.append(line)
                line = []
        else:
            line.append(token)
        
    return '\n'.join([' '.join([line for line in lines])])
           

import subprocess
from io import StringIO
import tokenize
def py_formatter(file_path):
    """
    Returns 'source' minus comments and docstrings and format.
    """
    try:
        subprocess.run(["black", file_path])
    except Exception as e:
        print(f"Error formatting code: {e}")
    with open(file_path, 'r', encoding='utf-8') as f:
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
    untok_s = "\n".join(lines)
    return untok_s