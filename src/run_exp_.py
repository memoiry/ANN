from tester import *
from utils import *
from test_func import *


if __name__ == '__main__':
	#print ' a)  ANN_GIST1M dataset using LSH algorithm'
	#gist_exp()
	print ' b)  ANN_GIST1M dataset using flann-kdtree algorithm'
	test_pyflann_gist()