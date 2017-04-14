import numpy as np
def vecs_read(filename, c_contiguous=True):
	temp = filename.split('.')
	#print 'tes'
	if temp[-1] == 'fvecs':
		#print 1
		fv = np.fromfile(filename, dtype=np.float32)
	else:
		#print 2
		fv = np.fromfile(filename, dtype=np.int32)
	if fv.size == 0:
		return np.zeros((0, 0))
	dim = fv.view(np.int32)[0]
	assert dim > 0
	fv = fv.reshape(-1, 1 + dim)
	if not all(fv.view(np.int32)[:, 0] == dim):
		raise IOError("Non-uniform vector sizes in " + filename)
	fv = fv[:, 1:]
	if c_contiguous:
		fv = fv.copy()
	return fv

def recall(result, ground_truth, k):
	recall_ = []
	count = 0
	if result.ndim == 1:
		res = [x in ground_truth for x in result]
		recall_.append(np.sum(res)/float(len(res)))
	else:
		for i in range(result.shape[0]):
			#count = count + 1
			#progress = float(count) / result.shape[0]
			#sys.stdout.writelines(str(progress) + '%' + '\r') 
			single_result = result[i][:k]
			single_ground = ground_truth[i][:k]
			res = [x in single_ground for x in single_result]
			recall_.append(np.sum(res)/float(len(res)))
	return np.mean(recall_)*100

def linear_search(dataset, query, k):
	num = query.shape[0]
	res = []
	for i in range(num):
		data = query[i]
		ans = [(ix, np.linalg.norm(data-x)) for ix, x in enumerate(dataset)]
		ans.sort(key=itemgetter(1))
		res.append(ans[:k])
	idx = []
	for i in range(num):
		temp = [x[0] for x in res[i]]
		idx.append(temp)
	idx = np.array(idx)
	return idx

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def run_time():
	global query, k_num, lsh, ground_truth
	result, dists = lsh.query(query, k_num)
	print recall(result, ground_truth, k_num)
