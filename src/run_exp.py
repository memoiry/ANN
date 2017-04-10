from tester import *
from utils import *
from test_func import *


if __name__ == '__main__':
	print ' a)  ANN_SIFT1M dataset using p-stable LSH algorithm'
	sift_exp()
	print ' b)  ANN_SIFT1M dataset using flann-kdtree algorithm'
	test_pyflann_sift()
	print ' c)  ANN_GIST1M dataset using p-stable LSH algorithm'
	gist_exp()
	print ' d)  ANN_GIST1M dataset using flann-kdtree algorithm'
	test_pyflann_gift()