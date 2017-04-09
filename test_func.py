from utils import *
import numpy as np

def random_test():
	dataset = np.random.randn(10000,100)*100
	testset = np.random.randn(100,100)*10000
	w = 4
	k_num = 5
	k_set = [6, 12]
	L_set = [12, 16, 20]
	ground_truth = linear_search(dataset, testset, k_num)
	print "ground truth computing finished"
	print "k L    recall    touched    recall/touched"
	for k in k_set:
		for L in L_set:
			lsh = PLSH(k, L, 5000)
			lsh.build_index(dataset)
			result, dists, candi_avg = lsh.query(testset, k_num)
			acc = recall(result, ground_truth, k_num)
			candi_avg = candi_avg * 100
			prop = acc / candi_avg
			print k, L, ' ',' ','{} %   {} %'.format(acc,candi_avg), ' ', ' ', prop

def random__test():
	d = 20
	xmax = 20
	num_points = 1000
	points = [[np.random.randint(0,xmax) for i in xrange(d)] for j in xrange(num_points)]
	num_neighbours = 2
	radius = 0.1
	for point in points[:num_points]:
		for i in xrange(num_neighbours):
			points.append([x+np.random.uniform(-radius,radius) for x in point])
	dataset = np.array(points)
	testset = np.array(points[:num_points/10])
	#print testset.shape
	k_num = 5
	ground_truth = linear_search(dataset, testset, k_num)
	print "Ground Truth Fished"
	print "k L    recall    touched    recall/touched"
	w_set = np.array([1, 2, 4, 8, 12, 14, 20])
	k_ = [2,4,8,12,16,20]
	L_ = [2,4,8,16,20]
	for k in k_:
		for L in L_:
			lsh = PLSH(k,L,1)
			lsh.build_index(dataset)
			result, dists, candi_avg = lsh.query(testset, k_num)
			#print "truth"
			#print np.sum(result == ground_truth) / 20 /100.0
			#print ' '
			acc = recall(result, ground_truth, k_num)
			candi_avg = candi_avg * 100
			prop = acc / candi_avg
			print k, L, ' ',' ','{} %   {} %'.format(acc,candi_avg), ' ', ' ', prop



def sift_test():
	#global query, k_num, lsh, ground_truth
	dataset = vecs_read('sift_base.fvecs')
	#dataset = dataset[:500000]
	query = vecs_read('sift_query.fvecs')
	query = query[:20]
	ground_truth = vecs_read('sift_groundtruth.ivecs')
	ground_truth = ground_truth[:100]
	print query.shape
	k_num = 50
	w = [15,18,20]
	k = 12
	L = 6
	#w = np.max(query) * 30 
	print "w k L    recall    touched    recall/touched"
	lsh = PLSH(k,L,w)
	lsh.build_index(dataset)
	result, dists, candi_avg = lsh.query(query, k_num)
	print "query finished"
	acc = recall(result, ground_truth, k_num)
	candi_avg = candi_avg * 100
	prop = acc / candi_avg
	print w, k, L, ' ',' ','{} %   {} %'.format(acc,candi_avg), ' ', ' ', prop
	##cProfile.run('run_time()', 'result')
	#p = pstats.Stats("result")
	#p.strip_dirs().sort_stats("cumulative").print_stats(10)
	#result, dists = lsh.query(query, k_num)
	#print recall(result, ground_truth, k_num)


def sift_exp():
	query_num = 20
	dataset = vecs_read('sift_base.fvecs')
	query = vecs_read('sift_query.fvecs')
	num = query.shape[0]
	rand_index = np.random.randint(0,num,size=(query_num))
	query = query[rand_index]
	ground_truth = vecs_read('sift_groundtruth.ivecs')
	ground_truth = ground_truth[rand_index]
	k_num = 100
	k = [14,16,20]
	L = [6,8,12]
	trees_num = [1,2,4,8,16]
	k_standard = 12
	L_standard = 6
	w_standard = 9
	trees_num_standard = 4
	lsh_result = []
	for k_ in k:
		tester = Tester(k_, L_standard, w_standard, trees_num_standard, dataset, query, k_num, ground_truth)
		lsh_result.append(tester.run())
	for L_ in L:
		tester = Tester(k_standard, L_, w_standard, trees_num_standard, dataset, query, k_num, ground_truth)
		lsh_result.append(tester.run())
	#for w_ in L:
	#	tester = Tester(k_standard, L_standard, w_, trees_num_standard, dataset, query, k_num, ground_truth)
	#	lsh_result.append(tester.run())
	lsh_result = np.array(lsh_result)
	np.savetxt("result.csv", lsh_result, delimiter = ",")
	#for trees_num_ in trees_num:
	#	tester = Tester(k_standard, L_standard, w_standard, trees_num_, dataset, query, k_num, ground_truth)
	#	lsh_result.append(tester.run())

def test_pyflann():
	dataset = vecs_read('sift_base.fvecs')
	query = vecs_read('sift_query.fvecs')
	#query = query[:1000]
	ground_truth = vecs_read('sift_groundtruth.ivecs')
	#ground_truth = ground_truth[:1000]
	k_num = 100
	flan = pyflann.FLANN()
	tree = [1,2,4,8,16,20]
	print "tree_num\trecall"
	for tree_num in tree:
		#print "building the index"
		params = flan.build_index(dataset, algorithm = "kdtree", trees = tree_num)
		#print params
		#print "index built"
		result, dists = flan.nn_index(query, k_num)
		acc = recall(result, ground_truth, k_num)
		print tree_num, ' ',' ','{} %'.format(acc)

def test_lsh():
	dataset = vecs_read('sift_base.fvecs')
	#dataset = dataset[:100000]
	query = vecs_read('sift_query.fvecs')
	query = query[:1000]
	ground_truth = vecs_read('sift_groundtruth.ivecs')
	ground_truth = ground_truth[:1000]
	k_num = 100
	key_size = 6
	table_num = 1
	lsh = LSHash(key_size, dataset.shape[1], table_num)
	count = 0
	for i in range(dataset.shape[0]):
		count = count + 1
		if (float(count)/dataset.shape[0]) % 0.01 == 0:
			print "progress: {} %".format(float(count)/dataset.shape[0]*100)
		lsh.index(dataset[i],i)
	result = []
	print "index finished"
	count = 0
	for i in range(query.shape[0]):
		count += 1
		print "progress: {} %".format(float(count)/query.shape[0]*100)
		single_query = query[i]
		temp = lsh.query(single_query, k_num)
		temp = [x[0] for x in temp] 
		result.append(temp)
	result = np.array(result)
	acc = recall(result,ground_truth ,k_num)
	print acc