
pre = ['dev', 'train']
fs = ['context', 'question']
for p in pre:
    for f in fs:
        print ('Processing ' + p + '.' +f)
        with open('../../data/' + p + '.' + f) as ff:
            content = ff.readlines()
        with open('../../data/' + p + '.' + f + '.pos') as ff_pos:
            pos_content = ff_pos.readlines()
        with open('../../data/' + p + '.' + f + '.ne') as ff_ne:
            ne_content = ff_ne.readlines()
        f_size = len(content)
        pos_size = len(pos_content)
        ne_size = len(ne_content)
        if f_size != pos_size or f_size != ne_size or pos_size != ne_size:
            print ('size mismatch!')
            exit()
        for i in range(f_size):
            f_line = content[i].split()
            pos_line = pos_content[i].split()
            ne_line = ne_content[i].split()
            if len(f_line) != len(pos_line) or len(f_line) != len(ne_line) or len(pos_line) != len(ne_line):
                print ('size mismatch!')
                exit()
