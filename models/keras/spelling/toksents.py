#!/usr/bin/env python

import sys
import os
from data import data
import marshal

sent_file = sys.argv[1]
d = data.load_data(sent_file)
token_seq = data.tokenize(d)
marshal_file = os.path.splitext(sent_file)[0] + '.marshal'
marshal.dump(token_seq, open(marshal_file, 'w'))
print('DONE ' + sent_file)
