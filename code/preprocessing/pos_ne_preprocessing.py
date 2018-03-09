# preprocess context, question
# generate pos tag and ne tags
import sys
sys.path.append('../')
from data_batcher import token_to_pos_ne_id

pos_tag_id_map = {}
with open('../../pos_tags.txt') as f:
    pos_tag_lines = f.readlines()
for i in range(len(pos_tag_lines)):
    pos_tag_id_map[pos_tag_lines[i][:-1]] = i + 1 # need to get rid of the trailing newline character
# get the NE tag to id
ne_tag_id_map = {}
all_NE_tag = ['B-FACILITY', 'B-GPE', 'B-GSP', 'B-LOCATION', 'B-ORGANIZATION', 'B-PERSON', 'I-FACILITY', 'I-GPE', 'I-GSP', 'I-LOCATION', 'I-ORGANIZATION', 'I-PERSON','O'] # I know this not elegant
for i in range(len(all_NE_tag)):
    ne_tag_id_map[all_NE_tag[i]] = i + 1

pre = ['dev', 'train']
fs = ['context', 'question']
for p in pre:
    for f in fs:
        print ('Processing ' + p + '.' +f)
        with open('../../data/' + p + '.' + f) as ff:
            content = ff.readlines()
        f_pos_out = open('../../data/' + p + '.' + f + '.pos', 'w')
        f_ne_out = open('../../data/' + p + '.' + f + '.ne', 'w')
        print ('file line size: ' + str(len(content)))
        for line in content:
            pos_result, ne_result = token_to_pos_ne_id(line.split(), pos_tag_id_map, ne_tag_id_map)
            pos_file_str = ""
            ne_file_str = ""
            for i in range(len(pos_result)):
                pos_file_str += str(pos_result[i]) + ' '
                ne_file_str += str(ne_result[i]) + ' '
            pos_file_str = pos_file_str[:-1] + '\n'
            ne_file_str = ne_file_str[:-1] + '\n'
            f_pos_out.write(pos_file_str)
            f_ne_out.write(ne_file_str)
        f_pos_out.close()
        f_ne_out.close()
