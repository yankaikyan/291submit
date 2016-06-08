# coding: utf8
import operator
import string
import collections
import os
import numpy as np
import tensorflow as tf

#poem_files = ["jueju5.txt", "jueju7.txt", "lvshi5.txt", "lvshi7.txt"]
poem_files = ["lvshi7.txt"]
index_chars_file = 'index_chars.txt'
path = './data/'
POEM_LEN = 66

# only keep the first vocab_size chars in the dictionary
# index start from start_index
# have 0:*, 1:^, 2:$
def getDicts(vocab_size):
  index_to_char = dict()
  char_to_index = dict()

  with open(path + index_chars_file,'r') as in_f:
    i = 0
    for line in in_f:
      entry = line.split(':')        
      if entry[1].strip():
        c = entry[1].strip().decode('utf8')
        index_to_char[i] = c
        char_to_index[c] = i
        i += 1
      if i >= vocab_size:
        break
  in_f.close()
  return index_to_char, char_to_index

#line is the input string in utf8, char_to_index is the dictionary
#add a ^ at the beginning, add $ until the length is POEM_LEN
#for chars not in the dictionary, use *:0 instead
def str_to_array(line, char_to_index):
  line = line.strip().decode('utf8')
  a = [1]
  for c in line:
    a.append(char_to_index.get(c, 0))
  while(len(a) < POEM_LEN):
    a.append(2)
  return a
    
#return a np.array of all the poems in the poem_files
#for chars not in char_to_index, use '*' instead (index as 0)
def read_poems(char_to_index):
  poem_list = []
  for f in poem_files:
    with open(path + f, 'r') as in_f:
      for line in in_f:
        poem_list.append(str_to_array(line, char_to_index))
    in_f.close()
  return np.array(poem_list)

                                
"""  
if __name__ == '__main__':
  index_to_char, char_to_index = getDicts(2000)
  poems = read_poems(char_to_index)
  print "finish read"
"""

