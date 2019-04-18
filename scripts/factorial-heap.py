import time
import sys
from sys import getsizeof

if len(sys.argv) != 2:
	print("USO {} [num_to_factorial]".format(sys.argv[0]))
	quit()


def start_generate(n):
    global result,count,total
    a = [str(i) for i in range(n)]
    start = time.time()
    generate(n,a)
    append_to_file('teste.txt')
    end = time.time()
    total = total + count
    result = list()
    print('Total of permutations: {}'.format(total))
    print('Time to process: {}'.format(end-start))


def generate(n,a):
    global result,count,total
    if n == 1:
        result.append(';'.join(s for s in a) + '\n')
        count = count + 1
        if getsizeof(result) >= 50000000:
            append_to_file('teste.txt')
            total = total + count
            count = 0
            print('#Size of result: {}, total: {}'.format(getsizeof(result), total))
            result = list()
    else:
        for i in range(n):
            generate(n-1,a)
            t = a[n-1]
            if n % 2 == 0:
                a[n-1] = a[i]
                a[i] = t
            else:
                a[n-1] = a[0]
                a[0] = t


def append_to_file(path):
    global result
    with open(path,'a') as f:
        for record in result:
            f.write(record)


result = list()
count,total = 0,0
start_generate(int(sys.argv[1]))