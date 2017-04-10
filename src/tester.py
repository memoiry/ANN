from collections import defaultdict
import pyflann
import numpy as np
from operator import itemgetter
import datetime as dt
import sys
import cProfile
import pstats  
import math
import utils
#from lshash import LSHash
E2LSH_BIGPRIMENUM = 4294967291
MAX_HASHRAND = 536870912

class PLSH:

	def __init__(self,k,L,w):
		self.w = w
		self.k = k
		self.L = L
		self.hash_tables = []

	def build_index(self, dataset):
		#dataset = normalized(dataset, 0)
		self.dataset = dataset
		self.d = dataset.shape[1]
		#print self.d
		self.hashfamily = self.build_hash_family()
		self.build_hash_tables()
		#for debug
		num = dataset.shape[0]
		times = len(self.hash_tables)
		count = 0
		#build index
		flag = 0
		for inde, (hash_funcs, table) in enumerate(self.hash_tables):
			for ix, p in enumerate(dataset):
				count = count + 1
				progress = float(count) / (num*times) * 100
				#if int(progress) > flag:
				#	flag += 1
				#	sys.stdout.writelines(str(progress) + "%" + '\r')
				sys.stdout.writelines(str(progress) + "%" + '\r')
				table[self.hash(hash_funcs, p, inde)].append(ix)
		#print ' '
	def build_hash_family(self):
		hash_family = []
		for i in range(self.L):
			hash_family.append(L2HashFamily(self.d, self.w, self.k))
		return hash_family

	def build_hash_tables(self):
		for i in range(self.L):
			temp = []
			for j in range(self.k):
				temp.append(self.hashfamily[i].build_hash_func())
			#print temp
			self.hash_tables.append((temp, defaultdict(lambda:[])))

	def l2_norm(self, x, y):
		return np.sqrt(np.sum((x-y)**2))

	def query(self, query_dataset, k):
		#query_dataset = normalized(query_dataset, 0)
		num = query_dataset.shape[0]
		res = []
		candi = []
		#count = 0
		#print ' '
		for i in range(num):
			#count = count + 1
			#print count 
			#progress = float(count) / num * 100
			#print progress
			#sys.stdout.writelines(str(progress) + '%' + '\r') 
			data = query_dataset[i,:]
			temp, candi_len = self.single_query(data, k)
			candi.append(candi_len)
			res.append(temp)
		ix, dists = self.wrap(res,num,k)
		ix = np.array(ix)
		candi = np.array(candi)
		candi_avg = np.mean(candi) / self.dataset.shape[0]
		dists = np.array(dists)
		return (ix, dists, candi_avg)

	def hamming_dist(self, x, y):
		num = len(x)
		count = 0
		for i in range(num):
			if x[i] != y[i]:
				count = count + 1
		return count

	def single_query(self, query_, k):
		candidates = set()
		for ix, (hash_funcs, table) in enumerate(self.hash_tables):
			#print self.hash(hash_funcs, query_)
			#query_hash = eval(self.hash(hash_funcs, query_))
			#print query_hash
			#compare_hamming = []
			#for hash_value, indx in table.iteritems():
			#	hash_value = eval(hash_value)
		#		dist_ = self.hamming_dist(query_hash, hash_value)
			#	compare_hamming.append((dist_, indx))
			#compare_hamming.sort(key=itemgetter(0))
			#matchs = []
			#for i in range(len(compare_hamming)):
			#	matchs.extend(compare_hamming[i][1])
			#print matchs[:k]
			key_ = self.hash(hash_funcs, query_, ix)
			matchs = table.get(key_, [])
			candidates.update(matchs)
			#print "pay attention"
			#print key_
			#print self.k
			for i in range(self.k):
				temp = key_[i]
				temp = '0' if temp == '1' else '1'
				neibour_key = key_[:i] + temp + key_[i:]
				matchs = table.get(neibour_key, [])
				candidates.update(matchs)
			#candidates.update(matchs[:k])
		#print candidates
		candi_len = len(candidates)
		#print float(candi_len) / self.dataset.shape[0]
		res = []
		#print candidates
		for ix in candidates:
			res.append((ix, self.l2_norm(self.dataset[ix,:], query_)))
		res.sort(key=itemgetter(1))
		return res[:k], candi_len

	def wrap(self, res, num, k):
		#print res
		ix = []
		dists = []
		for i in range(num):
			temp_ix = []
			temp_dists = []
			if res[i] != []:
				for j in res[i]:
					temp_ix.append(j[0])
					temp_dists.append(j[1])
			while len(temp_ix) < k:
				temp_ix.append(-1)
			while len(temp_dists) < k:
				temp_dists.append(-1)
			ix.append(temp_ix)
			dists.append(temp_dists)
		return (ix, dists)

	def hash(self, hash_funcs, p, ix):
		#print [h.hash(p) for h in hash_funcs]
		return self.hashfamily[ix].combine([h.hash(p) for h in hash_funcs])

class L2HashFamily:

	def __init__(self, d, w, k):
		self.d = d 
		self.w = w 
		self.r1 = np.random.randint(0,MAX_HASHRAND,size=(k))
		self.r2 = np.random.randint(0,MAX_HASHRAND,size=(k))

	def build_hash_func(self):
		return L2Hash(self.rand_vec(), self.rand_offset(), self.w)

	def rand_vec(self):
		return np.random.randn(self.d)

	def rand_offset(self):
		return np.random.uniform(0, self.w)

	def combine(self, table):
		#pair_ = 1 << np.arange(len(table) - 1, -1, -1)
		#print np.dot(table, pair_)
		#H1 = np.dot(table, self.r1)
		#H2 = np.dot(table, self.r2)
		# print H1
		#H1 = H1 // E2LSH_BIGPRIMENUM
		#H2 = H2 // E2LSH_BIGPRIMENUM
		#print (H1,H2)
		ans = ''
		for bir in table:
			ans += str(bir)
		#print ans
		return ans
		#print table
		#return str(table)

class L2Hash:

	def __init__(self, a, b, w):
		self.a = a 
		self.b = b
		self.w = w

	def hash(self, v):
		#print (np.dot(self.a, v) + self.b) // self.w
		#return int((np.dot(self.a, v) + self.b) / self.w)
		temp = 1 if np.dot(self.a, v) > 0 else 0
		return temp

class Tester:
	def __init__(self, k, L, w, trees_num, dataset, query, k_num, ground_truth):
		self.k = k
		self.L = L 
		self.w = w
		self.ground_truth = ground_truth
		self.dataset = dataset
		self.query = query
		self.k_num = k_num
		#self.trees_num = trees_num
		#self.flann_kdtree = pyflann.FLANN()
		self.lsh = PLSH(k, L, w)

	def run(self):
		# kdtree results
		#print("  a) flann kd-tree with {} trees".format(self.trees_num))
		#kdtree_results = []
		#start = dt.datetime.now()
		#self.flann_kdtree.build_index(self.dataset, algorithm="kdtree", trees = self.trees_num)
		#end = dt.datetime.now()
		#kdtree_results.append(end - start)

		#start = dt.datetime.now()
		#result, dists = self.flann_kdtree.nn_index(self.query, self.k_num)
		#end = dt.datetime.now()
		#kdtree_results.append(end - start)

		#acc = recall(result, self.ground_truth, self.k_num)
		#kdtree_results.append(acc)
		#self.print_result(kdtree_results)

		#lsh results
		print(" LSH with l2-norm metrics, k: {}, L: {}".format(self.k, self.L))
		lsh_results = []
		start = dt.datetime.now()
		self.lsh.build_index(self.dataset)
		end = dt.datetime.now()
		lsh_results.append((end - start).total_seconds())

		start = dt.datetime.now()
		result, dists, candi_avg = self.lsh.query(self.query, self.k_num)
		end = dt.datetime.now()
		lsh_results.append((end - start).total_seconds()/20)


		acc = recall(result, self.ground_truth, self.k_num)
		candi_avg = candi_avg * 100
		prop = acc / candi_avg
		lsh_results.append(acc)
		lsh_results.append(candi_avg)
		lsh_results.append(prop)
		self.print_result_lsh(lsh_results)
		return lsh_results


	def print_result(self, results):

	    stats = [("Build Index Time",     results[0]),
	           ("Query Time",  results[1]),
	           ("Accuracy", results[2])]
	    for s in stats:
	        print("    * {:<20} {:<20}".format(s[0] + ":", str(s[1])))
	    print("")
	    
	def print_result_lsh(self, results):
	    stats = [("Build Index Time",     results[0]),
	           ("Query Time",  results[1]),
	           ("Accuracy", results[2]),
	           ("touched", results[3]),
	           ("proportion", results[4])]
	    for s in stats:
	        print("    * {:<20} {:<20}".format(s[0] + ":", str(s[1])))
	    print("")


