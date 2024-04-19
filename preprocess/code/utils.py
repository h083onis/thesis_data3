from spiral import ronin

def out_code_dict(diff_text):
    is_start = False
    code_dict = {}
    code_dict['added_code'] = []
    code_dict['deleted_code'] = []
     
    lines = diff_text.split('\n')
    for line in lines[:-1]:
        if line[0:2] == '@@':
            is_start = True
        if is_start == True and line[1:].strip() != '': 
            if line[0] == '+': 
                code_dict['added_code'].append(line[1:])
            elif line[0] == '-':
                code_dict['deleted_code'].append(line[1:])
        
    return code_dict


def out_piece_snippet(diff_text):
    lines_extraction = []
    integrate_text = ''
    idx_list1 = []
    identifier_list = []
    is_start = False
    cnt = 0
    
    lines = diff_text.split('\n')
    for line in lines[:-1]:
        if line[0:2] == '@@':
            is_start = True
            idx_list1.append(cnt)
        if is_start == True and(line[0] == '+' or line[0] == ' '):
            tmp = line[1:].strip()
            if tmp.isidentifier() == True:
                identifier_list = ronin.split(tmp)
                for identity in identifier_list:
                    lines_extraction.append(identity.lower())
                cnt += len(identifier_list)
            else:
                lines_extraction.append(line[1:].strip())
                cnt += 1
            
    n = 1
    integrate_list = []
    idx_list1.append(len(lines_extraction))
    idx_list2 = idx_list1[n:] + idx_list1[:n]
    for idx1, idx2 in zip(idx_list1[:-1], idx_list2[:-1]):
        integrate_text = '\t'.join(lines_extraction[idx1:idx2])
        integrate_list.append(integrate_text)
        
    return integrate_list


def out_snippet_to_txt(filename, idx, snippet_list):
    integrate_txt = ''
    cnt = 0
    with open(filename, 'a', encoding='utf-8') as f_snippet:
        for snippet in snippet_list:
            cnt += 1
            integrate_txt = idx + '-'+str(cnt) +'\t' + snippet
            print(integrate_txt, file=f_snippet)
    

def out_txt(filename, text):
    with open(filename, 'w', encoding='utf-8', errors='ignore') as f:
        f.truncate(0)
        f.seek(0)
        text_list = text.split('\r\n')
        for text in text_list:
            print(text, file=f)