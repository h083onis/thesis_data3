import sentencepiece as spm
import json
import pandas as pd
import argparse


class SpmProcessor():
    def __init__(self, args=None):
        self.args = args
        
    def train(self, input_list):
        spm.SentencePieceTrainer.Train(
            sentence_iterator = iter(input_list),
            model_type = self.args.model_type,
            model_prefix = self.args.model_prefix,
            vocab_size = self.args.vocab_size,
            character_coverage = self.args.character_coverage,
            user_defined_symbols = self.args.user_defined_symbols,
        )
        
    def initializer(self, model_file):
        self.spm = spm.SentencePieceProcessor(model_file=model_file)
    
    def encode(self, line, out_type=int):
        encode = self.spm.Encode(line, out_type=out_type)
        return encode

    def decode(self, line):
        decode = self.spm.Decode(line)
        return decode
    
    def generate_vocab(self):
        vocab = {token:i for i, token in enumerate([self.spm.id_to_piece(id) for id in range(self.spm.get_piece_size())])}
        return vocab
    
def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-json_file', type=str)
    parser.add_argument('-model_type', type=str, default='unigram')
    parser.add_argument('-model_prefix', type=str, default='spm')
    parser.add_argument('-vocab_size', type=int, default=4000)
    parser.add_argument('-character_coverage', type=int, default=1.0)
    parser.add_argument('-user_defined_symbols', type=list, default=[])
    return parser
    
if __name__ == '__main__':
    params = read_args().parse_args()
    processor = SpmProcessor(params)
    with open(params.json_file, 'r') as f:
        data = json.load(f)
    kind_list = ['added_code', 'deleted_code']
    lines = []
    for commit in data:
        if commit['codes'] != []:
            for file in commit['codes']:
                for kind in kind_list:
                    if file[kind] != []:
                        lines += file[kind]
    processor.train(lines)