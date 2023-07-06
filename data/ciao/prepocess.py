import os

train_file = os.getcwd() + '/train_list.txt'
output_file = os.getcwd() + '/train.txt'

contents = ''

with open(train_file) as f:
    for l in f.readlines():
        l = l.strip('\n')
        real = [i for i in l.split('\t')[1:]]
        real_contents = ' '.join(real)
        contents += real_contents +'\n'


# with open(train_file) as f:
#     for l in f.readlines():
#         l = l.replace('\t', ' ')
#         contents += l

with open(output_file, 'w') as f:
    f.write(contents)




