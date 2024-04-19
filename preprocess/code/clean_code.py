import re
from spiral import ronin
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.lexers import JavaLexer
from pygments.lexers import CLexer
from pygments.lexers import CppLexer
from pygments.token import Token

class CleanCode():
    def __init__(self):
        pass
    def excute(self, line, filepath=None):
        clean_line = line
        clean_line = clean_line.replace('\r', "")
        if filepath != None:
            clean_line = self.tokenize_code(clean_line, filepath)
        clean_line = self.lower_line(clean_line)
        clean_line = self.clean_waste_space(clean_line)
        # clean_line = self.replace_num2sp(clean_line)
        return clean_line
    
    def clean_waste_space(self, line):
        line = [token for token in line.split(' ') if token != '']
        return ' '.join(line)

    def lower_line(self, line):
        return line.lower()
    
    # def replace_num2sp(self, line):
    #     line = ['<num>' if self.is_float(w) else '<num>' if self.is_integer(w) else w for w in line]
    #     return line
    
    def is_hexadecimal(self,value):
        try:
            int(value, 16)
            return True
        except ValueError:
            return False

    def is_float(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def is_integer(self, value):
        try:
            int(value)
            return True
        except ValueError:
            return False
        
    def tokenize_code(self, code, filepath):
        ext = filepath.split('.')[-1].lower()
        if ext == 'py': 
            tokens = list(lex(code, PythonLexer()))
        elif ext == 'cpp' or ext == 'hpp' or ext == 'cxx' or ext == 'hxx':
            tokens = list(lex(code, CppLexer()))
        elif ext == 'c' or ext == 'h':
            tokens = list(lex(code, CLexer()))
        elif ext == 'java':
            tokens = list(lex(code, JavaLexer()))
        
        code_list = []
        for token in tokens:
            if token[0] in Token.Literal:
                code_list.append('<literal>')
            elif token[0] in Token.Name:
                code_list.extend(['<num>'if tmp.isnumeric() else tmp.strip().lower() for tmp in ronin.split(token[1])])
            elif token[0] in Token.Text:
                continue
            elif token[0] in Token.Keyword:
                code_list.append(token[1])
            else:
                tmp = token[1].strip().lower()
                code_list.append('<num>' if self.is_float(tmp) else '<num>' if self.is_integer(tmp) else '<num>' if self.is_hexadecimal(tmp) else tmp)
            # code_list = ['<num>' if is_float(w) else '<num>' if is_integer(w) else '<num>' if is_hexadecimal(w) else w for w in code_list]
            # code_list = ['<num>' if is_float(w) else '<num>' if is_integer(w) else w for w in code_list]
        return ' '.join(code_list)
    