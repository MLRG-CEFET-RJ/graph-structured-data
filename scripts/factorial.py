import sys
import time

if len(sys.argv) != 2:
	print("USO {} [num_to_factorial]".format(sys.argv[0]))
	quit()

def grava(X,NN):
    num = ''
    for i in range(1,NN+1):
        #print(X[i])
        num = num + ';' + str(X[i])
    num = num[1:] + '\n'
    #print(num)
    #f = open('all_permut.txt', 'a')
    #f.write(num)
    #f.close()


def perm(S,K,N):
    saux = dict()
    if K>N:
        grava(S,N)
    # coloque k em todas as K posições possíveis e chama perm com K+1
    else:
        for i in reversed(range(1,K+1)):
            for j in reversed(range(1,K)):
                saux[j+1] = S[j]
            saux[i] = K
            for j in reversed(range(1,i)):
                saux[j]=S[j]
            perm(saux,K+1,N)

n = sys.argv[1]
s = dict()
s[1] = 1

start = time.time()
perm(s,2,int(n))
end = time.time()
print(end - start)
