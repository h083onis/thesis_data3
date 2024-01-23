from lang_processors.python_processor2 import PythonProcessor
import pickle
if __name__ == '__main__':
    processor = PythonProcessor()
    # processor.formatter('before2.py')
    # processor.formatter('after2.py')
    added_line_num, deleted_line_num = processor.diff(file_path='before2.py', file_path2='after2.py')
    pickle.dump([added_line_num, deleted_line_num], open('line_num.pkl', 'wb'))
    