from utils import *
import numpy as np
from tester import *
from memory_profiler import memory_usage

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
	dataset = vecs_read('../data/sift/sift_base.fvecs')
	#dataset = dataset[:500000]
	query = vecs_read('../data/sift/sift_query.fvecs')
	query = query[:20]
	ground_truth = vecs_read('../data/sift/sift_groundtruth.ivecs')
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

def gist_exp():
	query_num = 20
	dataset = vecs_read('../data/gist/gist_base.fvecs')
	query = vecs_read('../data/gist/gist_query.fvecs')
	print 3
	#dataset = vecs_read('/volumes/seagate backup plus drive/gist/gist_base.fvecs')
	#query = vecs_read('/volumes/seagate backup plus drive/gist/gist_query.fvecs')
	#query = query[:1000]
	#ground_truth = vecs_read('/volumes/seagate backup plus drive/gist/gist_groundtruth.ivecs')

	num = query.shape[0]
	rand_index = np.random.randint(0,num,size=(query_num))
	query = query[rand_index]
	ground_truth = vecs_read('../data/gist/gist_groundtruth.ivecs')
	ground_truth = ground_truth[rand_index]
	k_num = 100
	k = [14,16]
	L = [6,8]
	#trees_num = [1,2,4,8,16]
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

def sift_exp():
	query_num = 20
	dataset = vecs_read('../data/sift/sift_base.fvecs')
	query = vecs_read('../data/sift/sift_query.fvecs')
	num = query.shape[0]
	rand_index = np.random.randint(0,num,size=(query_num))
	query = query[rand_index]
	ground_truth = vecs_read('../data/sift/sift_groundtruth.ivecs')
	ground_truth = ground_truth[rand_index]
	k_num = 100
	k = [14,16,20]
	L = [6,8,12]
	#trees_num = [1,2,4,8,16]
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


def test_pyflann_sift():
	dataset = vecs_read('../data/sift/sift_base.fvecs')
	#print dataset
	query = vecs_read('../data/sift/sift_query.fvecs')
	#query = query[:1000]
	ground_truth = vecs_read('../data/sift/sift_groundtruth.ivecs')
	#ground_truth = ground_truth[:1000]
	k_num = 100
	tree = [1,4,8,16]
	checks_set = [16,32,64,128]
	res = []
	print "tree_num\tchecks\trecall\tbuild_time\tsearch_time"
	for tree_num in tree:
		for check in checks_set:
			flan = pyflann.FLANN(algorithm = "kdtree", trees = tree_num, checks = check)
			#print "building the index"
			start = dt.datetime.now()
			params = flan.build_index(dataset)
			end = dt.datetime.now()
			tim1 = (end-start).total_seconds()
			#print params
			#print "index built"
			start = dt.datetime.now()
			result, dists = flan.nn_index(query, k_num)
			end = dt.datetime.now()
			tim = (end-start).total_seconds()
			acc = recall(result, ground_truth, k_num)
			res.append([tree_num,check,acc,tim1,tim])
			print tree_num,'\t',check,'\t','{} %'.format(acc),'\t',tim1,'\t',tim
	np.savetxt('SIFT1M_FLANN.csv',np.array(res),delimiter = ",")

def test_pyflann_gist():
	#dataset = vecs_read('../data/gist/gist_base.fvecs')
	#query = vecs_read('../data/gist/gist_query.fvecs')
	#query = query[:1000]
	#ground_truth = vecs_read('../data/gist/gist_groundtruth.ivecs')

	dataset = vecs_read('/volumes/seagate backup plus drive/gist/gist_base.fvecs')
	query = vecs_read('/volumes/seagate backup plus drive/gist/gist_query.fvecs')
	#query = query[:1000]
	ground_truth = vecs_read('/volumes/seagate backup plus drive/gist/gist_groundtruth.ivecs')

	#ground_truth = ground_truth[:1000]
	k_num = 5
	tree = [1,4,8]
	checks_set = [32,64,128]
	res = []
	print "tree_num\tchecks\trecall\tbuild_time\tsearch_time"
	for tree_num in tree:
		for check in checks_set:
			flan = pyflann.FLANN(algorithm = "kdtree", trees = tree_num, checks = check)
			#print "building the index"
			start = dt.datetime.now()
			params = flan.build_index(dataset)
			end = dt.datetime.now()
			tim1 = (end-start).total_seconds()
			#print params
			#print "index built"
			start = dt.datetime.now()
			result, dists = flan.nn_index(query, k_num)
			end = dt.datetime.now()
			tim = (end-start).total_seconds()
			acc = recall(result, ground_truth, k_num)
			res.append([tree_num,check,acc,tim1,tim])
			print tree_num,'\t',check,'\t','{} %'.format(acc),'\t',tim1,'\t',tim
	np.savetxt('GIST1M_FLANN.csv',np.array(res),delimiter = ",")


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

def exp_flann_sift():
	global flan, dataset
	dataset = vecs_read('../data/sift/sift_base.fvecs')
	#print dataset
	query = vecs_read('../data/sift/sift_query.fvecs')
	query = query[:1000]
	ground_truth = vecs_read('../data/sift/sift_groundtruth.ivecs')
	#ground_truth = ground_truth[:1000]
	tree_num_set = [2,8,16,24]
	target_recall = 92
	k_num = 20
	print "tree_num\tchecks\trecall\tQueries per second"
	query_num = query.shape[0]
	res = []
	for tree_num in tree_num_set:
		checks = k_num
		recall_ = 0
		count = 0
		flan = pyflann.FLANN(algorithm = "kdtree", trees = tree_num, checks = checks)
		start = dt.datetime.now()
		mem_usage = memory_usage(build_index)
		end = dt.datetime.now()
		tim1 = (end-start).total_seconds()
		print 'Build index time: {} s'.format(tim1)
		print('Maximum memory usage: %s MB' % max(mem_usage))
		print 'Total memory usage: ', get_size(flan)/1024/1024,'MB'
		while recall_ < target_recall:
			count = count + 1
			start = dt.datetime.now()
			result, dists = flan.nn_index(query, k_num, checks = checks)
			end = dt.datetime.now()
			tim = (end-start).total_seconds()
			tim = query_num/tim
			recall_ = recall(result, ground_truth, k_num)
			res.append((recall_,tim,tree_num))
			checks = checks + 8 * count
			print tree_num,'\t{}\t{} %\t{}'.format(checks,recall_,tim)
	np.savetxt('result/SIFT1M_FLANN.csv',np.array(res),delimiter = ",")

def build_index():
	global flan, dataset
	flan.build_index(dataset)

def exp_flann_gist():
	global flan, dataset
	#dataset = vecs_read('../data/gist/gist_base.fvecs')
	#query = vecs_read('../data/gist/gist_query.fvecs')
	#ground_truth = vecs_read('../data/gist/gist_groundtruth.ivecs')
	dataset = vecs_read('/volumes/seagate backup plus drive/gist/gist_base.fvecs')
	query = vecs_read('/volumes/seagate backup plus drive/gist/gist_query.fvecs')
	query = query[:300]
	ground_truth = vecs_read('/volumes/seagate backup plus drive/gist/gist_groundtruth.ivecs')
	tree_num_set = [2,8,16,24]
	target_recall = 92
	k_num = 20
	print "tree_num\tchecks\trecall\tQueries per second"
	query_num = query.shape[0]
	res = []
	for tree_num in tree_num_set:
		checks = k_num
		recall_ = 0
		count = 0
		flan = pyflann.FLANN(algorithm = "kdtree", trees = tree_num, checks = checks)
		start = dt.datetime.now()
		mem_usage = memory_usage(build_index)
		end = dt.datetime.now()
		tim1 = (end-start).total_seconds()
		print 'Build index time: {} s'.format(tim1)
		print('Maximum memory usage: %s MB' % max(mem_usage))
		print 'Total memory usage: ', get_size(flan)/1024/1024,'MB'
		while recall_ < target_recall:
			count = count + 1
			start = dt.datetime.now()
			result, dists = flan.nn_index(query, k_num, checks = checks)
			end = dt.datetime.now()
			tim = (end-start).total_seconds()
			tim = query_num/tim
			recall_ = recall(result, ground_truth, k_num)
			res.append((recall_,tim,tree_num))
			checks = checks + 100 * count
			print tree_num,'\t{}\t{} %\t{}'.format(checks,recall_,tim)
	np.savetxt('result/GIST1M_FLANN.csv',np.array(res),delimiter = ",")

def exp_lsh_sift():
	global lsh, dataset
	dataset = vecs_read('../data/sift/sift_base.fvecs')
	query = vecs_read('../data/sift/sift_query.fvecs')
	ground_truth = vecs_read('../data/sift/sift_groundtruth.ivecs')
	query_num = 10
	num = query.shape[0]
	rand_index = np.random.randint(0,num,size=(query_num))
	query = query[rand_index]
	ground_truth = ground_truth[rand_index]
	k_num = 20
	w=1
	k_set = [8,12,16]
	L_set = [1,2,4,8,12]
	res = []
	print 'k\tL\trecall\tQueries per second\tBuild index time(s)\tmax memory(MB)\ttotal memory(MB)'
	for L in L_set:
		for k in k_set:
			lsh =  PLSH(k,L,w)
			start = dt.datetime.now()
			mem_usage = memory_usage(build_lsh)
			#print mem_usage
			end = dt.datetime.now()
			tim1 = (end-start).total_seconds()
			max_memory = max(mem_usage)
			total_memory = get_size(lsh)/1024/1024
			start = dt.datetime.now()
			result, dists, candi_avg = lsh.query(query, k_num)
			end = dt.datetime.now()
			tim = (end-start).total_seconds()
			tim = query_num/tim
			acc = recall(result, ground_truth, k_num)
			res.append((acc, tim,tim1,max_memory,total_memory))
			#print res
			print '{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(k,L,acc,tim,tim1,max_memory,total_memory)
	np.savetxt("result/SIFT1M_LSH.csv", res, delimiter = ",")

def build_lsh():
	global lsh, dataset
	lsh.build_index(dataset)

def exp_lsh_gist():
	global lsh, dataset
	#dataset = vecs_read('../data/gist/gist_base.fvecs')
	#query = vecs_read('../data/gist/gist_query.fvecs')
	#ground_truth = vecs_read('../data/gist/gist_groundtruth.ivecs')
	dataset = vecs_read('/volumes/seagate backup plus drive/gist/gist_base.fvecs')
	query = vecs_read('/volumes/seagate backup plus drive/gist/gist_query.fvecs')
	ground_truth = vecs_read('/volumes/seagate backup plus drive/gist/gist_groundtruth.ivecs')
	query_num = 10
	num = query.shape[0]
	rand_index = np.random.randint(0,num,size=(query_num))
	query = query[rand_index]
	ground_truth = ground_truth[rand_index]
	k_num = 20
	w=1
	k_set = [8,12,16]
	L_set = [1,2,4,8,12]
	res = []
	print 'k\tL\trecall\tQueries per second\tBuild index time(s)\tmax memory(MB)\ttotal memory(MB)'
	for L in L_set:
		for k in k_set:
			lsh =  PLSH(k,L,w)
			start = dt.datetime.now()
			mem_usage = memory_usage(build_lsh)
			end = dt.datetime.now()
			tim1 = (end-start).total_seconds()
			max_memory = max(mem_usage)
			total_memory = get_size(lsh)/1024/1024
			start = dt.datetime.now()
			result, dists, candi_avg = lsh.query(query, k_num)
			end = dt.datetime.now()
			tim = (end-start).total_seconds()
			tim = query_num/tim
			acc = recall(result, ground_truth, k_num)
			res.append((acc, tim, tim1, max_memory, total_memory))
			print '{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(k,L,acc,tim,tim1,max_memory,total_memory)
	np.savetxt("result/GIST1M_LSH.csv", res, delimiter = ",")