from tester import *
from utils import *
from test_func import *


if __name__ == '__main__':
	print ' a)  ANN_SIFT1M dataset using flann-kdtree algorithm'
	exp_flann_sift()
	print ' b)  ANN_SIFT1M dataset using p-stable LSH algorithm'
	exp_lsh_sift()
	print ' c)  ANN_GIST1M dataset using flann-kdtree algorithm'
	exp_flann_gist()
	print ' d)  ANN_GIST1M dataset using p-stable LSH algorithm'
	exp_lsh_gist()