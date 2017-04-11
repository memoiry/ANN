from tester import *
from utils import *
import sys


if __name__ == "__main__":
	k_num = int(sys.argv[1])
	k = int(sys.argv[3])
	L = int(sys.argv[2])
	dataset = vecs_read('dataset.fvecs')
	query = vecs_read('query.fvecs')
	lsh = PLSH(k,L,1)
	lsh.build_index(dataset)
	result, dists, candi_avg = lsh.query(query,k_num)
	np.savetxt("result.csv", result, delimiter = ",")
	np.savetxt("dists.csv", dists, delimiter = ",")