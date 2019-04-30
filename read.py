import os
from bloom_filter import BloomFilter
import fileinput
import sys
import pickle
from collections import Counter
# fastq-dump -X 500 -Z SRR1186053
# spots: 41,997,857
# name: SRR1186053



def one_hot_encoding(line):
    onehot = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    return [onehot[i] for i in line.replace('\n', '')]


def read_file(file_name):
    with open('out_f.txt', 'w') as output:
        # with open(file_name, "r") as text_file:
            # for i in range(1, 41997857*4, 4):
        counter = 0
        for line in fileinput.input([file_name]):
            if counter == 1:
                # print(len(line))
                output.write(line)
                # output.write(str(one_hot_encoding(line)))
                counter+=1
            elif counter == 3:
                counter = 0
            else:
                counter += 1

#     lines = text_file.read().split('\n')
# #     ids, GCs = reformat(lines[:-1])
#     text_file.close()
#     reads = [lines[i] for i in range(1, len(lines), 4)]
    pass

def get_kmers(read, k):
    return [read[i:i+k] for i in range(len(read)-k+1)]

all_data_path = 'D:/University_Work/Spring 2019/566/final/raw_data/'

# def count():
def split(all, train, valid, total):
    a, b = int(train*total), int(total*(train+valid))
    counter = 0
    with open('train.txt', 'w') as train_file, open('valid.txt', 'w') as valid_file, open('test.txt', 'w') as test_file:
        for line in fileinput.input([all]):
            if counter < a:
                train_file.write(line)
            elif a < counter < b:
                valid_file.write(line)
            else:
                test_file.write(line)
            counter += 1
    pass

def check_file(filename):
    for line in fileinput.input([filename]):
        print(line)

if __name__ == '__main__':
    # all_data = os.listdir(all_data_path)
    # test = all_data_path+all_data[0]
    # print(all_data_path+all_data[0])
    # seqs = read_file('sratoolkit.2.9.6-win64/bin/data.txt')
    # print(seqs)
    # read_file('sratoolkit.2.9.6-win64/bin/data.txt')
    # read_file('test.txt')
    total = 41997857
    # split('out_f.txt', 0.4, 0.4, total)
    # check_file('test.txt')
    seqs = open('out.txt', 'r')
    seqs = seqs.read().split('\n')
    # k = 10
    # x = -1.2
    # print(sys.getsizeof(x))
    # print(seqs[0])
    # kmers = []
    # for seq in seqs:
    #     kmers.append(get_kmers(seq, k))
    # for km in kmers:
    #     print(Counter(km))
    # print(get_kmers(seqs[0], k))
    # print(sys.getsizeof(kmers))

    bloom = BloomFilter(max_elements=16000, error_rate=0.05)
    bloom2 = BloomFilter(max_elements=6719656, error_rate=0.05)
    bloom.add(seqs[10])
    print(sys.getsizeof(bloom))
    # Test whether the bloom-filter has seen a key:
    with open('testfilter.bloom', 'wb') as testfilter:

        # Step 3
        pickle.dump(bloom, testfilter)

    with open('testfilter.bloom', 'rb') as testfilter:

        # Step 3
        bloom2 = pickle.load(testfilter)

        # After config_dictionary is read from file
        print(bloom2)

    assert seqs[0] not in bloom, "seq found before adding"
    # Mark the key as seen
    bloom.add(seqs[0])

    # Now check again
    assert seqs[0] in bloom, "seq not found"