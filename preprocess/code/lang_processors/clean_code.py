import tokenize
from io import BytesIO
from sacrebleu.tokenizers.tokenizer_intl import TokenizerV14International

class PythonProcesser2():
    def __init__(self):
        
        self.spetoken2char = {
            "STOKEN00": "#",
            "STOKEN1": "\\n",
            "STOKEN2": '"""',
            "STOKEN3": "'''",
        }
        
        self.char2spetoken = {
            value: "" + key + " " for key, value in self.spetoken2char.items()
        }
        self.language = "python"
        self.tokenize_string = TokenizerV14International()
        
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
                tokentype, token, _, _, line = next(iterator)
            except(
                    tokenize.TokenError,
                    IndentationError,
                    SyntaxError,
                    UnicodeDecodeError,
            ) as e:
                pass
            except StopIteration:
                # raise Exception(f"End of iterator before ENDMARKER token.")
                break
            
            
            if tokentype == tokenize.ENCODING or tokentype == tokenize.NL:
                continue
            
            elif tokentype == tokenize.NEWLINE:
                if removed_docstr == 1:
                    removed_docstr = 0
                    continue
                tokens.append("NEW_LINE")
                
            elif tokentype == tokenize.COMMENT: #one line comment
                continue
            
            elif tokentype == tokenize.STRING:
                if token == line.strip():  # docstring
                    removed_docstr = 1
                    continue
                else:
                    tokens.append(self.tokenize_string(token))
                    
            elif tokentype == tokenize.INDENT:
                tokens.append("INDENT")

            elif tokentype == tokenize.DEDENT:
                # empty block
                if tokens[-1] == "INDENT":
                    tokens = tokens[:-1]
                else:
                    tokens.append("DEDENT")

            elif tokentype == tokenize.ENDMARKER:
                tokens.append("ENDMARKER")
                break 
            
            else:
                tokens.append(token)

        # assert tokens[-1] == "ENDMARKER", "Error, no end marker"
        if tokens[-1] != "ENDMARKER":
            return tokens
        else:
            return tokens[:-1]
        