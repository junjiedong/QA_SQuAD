# read answer, context, question
# generate vocabulary file for elmo batcher
pre = ['dev', 'train']
fs = ['answer', 'context', 'question']
result = set()
for p in pre:
    for f in fs:
        print ('Processing ' + p + '.' +f)
        with open('../../data/' + p + '.' + f) as ff:
            content = ff.readlines()
        for line in content:
            for token in line.split():
                result.add(token)

ff = open('../../data/elmo_voca.txt', 'w')
max_length = 0
for item in result:
    if len(item) > max_length:
        max_length = len(item)
    ff.write(item + '\n')
ff.write('</S>\n')
ff.write('<S>\n')
ff.write('<UNK>\n')
ff.close()
print ('max length of the token in the data:' + str(max_length))
