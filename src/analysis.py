import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_flann_sift_result():
	tree_n = [1,2,4,8,16,20,50]
	tree_num = [[x] * 3 for x in tree_n]
	tree_num = np.array(tree_num).flatten()
	build_index_time = [2.185,2.051174,2.012371,4.11,4.289341,3.964041,
	8.21,8.134533,8.223954,17.200673,17.199582,
	17.01,33.255143,34.207272,33.07,41.901983,
	41.946438,40.68,111.675014,104.236176,100.87]
	search_time = [0.639,0.815477,0.765852,0.694938,0.784679,
	0.68,0.750577,0.82667,0.72,0.812683,0.94513,
	0.79,0.918106,0.998308,1.28,0.933258,1.020692,0.94,3.551357,1.365467,9.31]
	recall = [14.926,15.176,14.762,17.096,16.93,17.35,19.33,
	19.422,19.324,21.406,21.71,21.134,23.02,23.104,23.128,23.774,
	23.4,23.714,24.782,25.116,24.798]
	return tree_num, build_index_time, search_time, recall

def get_flann_gist_result():
	tree_n = [1,2,4,8,16,20]
	tree_num = [[x] * 3 for x in tree_n]
	tree_num = np.array(tree_num).flatten()
	build_index_time = [25.136294,21.139055,20.858293,15.770843,14.298888,
	14.923447,30.872005,28.827316,30.174561,
	60.361764,57.949673,58.830932,
	131.236339,113.808471,118.337752,143.578958,146.324523,152.398442]
	search_time = [0.208519,0.182428,0.197556,
	0.222337,0.193023,0.20096,
	0.2275,0.202142,0.19252,
	0.2748,0.230911,0.223402,
	0.253231,0.233742,0.263411,
	0.342395,0.306826,0.322426]
	recall = [1.659,1.704,1.735,1.911,1.934,1.946,2.172,2.097,2.122,
	2.395,2.334,2.407,2.668,2.637,2.666,2.664,2.707,2.727]
	return tree_num, build_index_time, search_time, recall

def get_lsh_sift_result():
	L = [6,8,12]
	k = [14,16,20]
	build_index_time = [327.57,439.08,662.57,357.82,404.68,511.31]
	search_time = [1.213,1.048,1.743,0.654,0.657,1.502]
	recall = [59.8,67.85,83.25,52.2,51.85,30.05]
	touched = [6.95, 5.82, 9.74,3.8,3.92,0.72]
	proportion = [8.60,11.65,8.54,13.71,13.19,41.99]
	return k, L, build_index_time, search_time, recall, touched, proportion

def get_lsh_gist_result():
	L = [6,8]
	k = [14,16]
	build_index_time = [407.17,564.53,359.06,502.13]
	search_time = [6.67,5.57,1.29,3.946]
	recall = [58.35,60.25,34.4,41.2]
	touched = [30.40,24.74,9.88,17.57]
	proportion = [1.92,2.43,3.269,2.35]
	return k, L, build_index_time, search_time, recall, touched, proportion


def box_plot(tree_num, build_index_time, search_time, recall, datatype):
	#print tree_num
	df_build = pd.DataFrame({'tree_num':tree_num,'build_index_time':build_index_time})
	#print df_build
	df_search = pd.DataFrame({'tree_num':tree_num,
	    'search_time':search_time})
	df_recall = pd.DataFrame({'tree_num':tree_num,
	    'recall':recall})
	fig = plt.figure(figsize=(15, 10))
	fig.add_subplot(131)
	f = sns.boxplot(x="tree_num", y = "build_index_time", data = df_build)
	f.set_xlabel('tree_num')
	f.set_ylabel('build_index_time')

	#plt.savefig('figure/{}_flann_build_time.png'.format(datatype))
	#plt.close(0)
	fig.add_subplot(132)
	f = sns.boxplot(x="tree_num", y = "search_time", data = df_search)
	#plt.savefig('figure/{}_flann_search_time.png'.format(datatype))
	#plt.close(0)
	f.set_xlabel('tree_num')
	f.set_ylabel('search_time')

	fig.add_subplot(133)
	f = sns.boxplot(x="tree_num", y = "recall", data = df_recall)
	f.set_xlabel('tree_num')
	f.set_ylabel('recall')
	fig.savefig('figure/{}_flann.png'.format(datatype))
	plt.close(0)

def linear_plot(k, L, build_index_time, search_time, recall, touched, proportion,datatype):
	#print tree_num
	midi = len(build_index_time)/2
	build_index_time_L = build_index_time[:midi]
	build_index_time_k = build_index_time[-midi:]
	search_time_L = search_time[:midi]
	search_time_k = search_time[-midi:]
	recall_L = recall[:midi]
	recall_k = recall[-midi:]
	touched_L = touched[:midi]
	touched_k = touched[-midi:]
	proportion_L = proportion[:midi]
	proportion_k = proportion[-midi:]
	df_L = pd.DataFrame({'L':L,'build_index_time':build_index_time_L,
		'search_time':search_time_L,'recall':recall_L,'touched':touched_L,
		'proportion':proportion_L})
	df_k = pd.DataFrame({'k':k,'build_index_time':build_index_time_k,
		'search_time':search_time_k,'recall':recall_k,'touched':touched_k,
		'proportion':proportion_k})
	plt.figure(figsize=(15, 15))
	sns.lmplot(x="L", y = "build_index_time", data = df_L, ci=None, scatter_kws={"s": 80})
	plt.savefig('figure/{}_L_build_time_lsh.png'.format(datatype))
	plt.close(0)
	plt.figure(figsize=(15, 15))
	sns.lmplot(x="L", y = "search_time", data = df_L, ci=None, scatter_kws={"s": 80})
	plt.savefig('figure/{}_L_search_time_lsh.png'.format(datatype))
	plt.close(0)
	plt.figure(figsize=(15, 15))
	sns.lmplot(x="L", y = "recall", data = df_L, ci=None, scatter_kws={"s": 80})
	plt.savefig('figure/{}_L_recall_lsh.png'.format(datatype))
	plt.close(0)
	plt.figure(figsize=(15, 15))
	sns.lmplot(x="L", y = "touched", data = df_L, ci=None, scatter_kws={"s": 80})
	plt.savefig('figure/{}_L_touched_lsh.png'.format(datatype))
	plt.close(0)
	plt.figure(figsize=(15, 15))
	sns.lmplot(x="L", y = "proportion", data = df_L, ci=None, scatter_kws={"s": 80})
	plt.savefig('figure/{}_L_proportion_lsh.png'.format(datatype))
	plt.close(0)
	plt.figure(figsize=(15, 15))
	sns.lmplot(x="k", y = "build_index_time", data = df_k, ci=None, scatter_kws={"s": 80})
	plt.savefig('figure/{}_k_build_time_lsh.png'.format(datatype))
	plt.close(0)
	plt.figure(figsize=(15, 15))
	sns.lmplot(x="k", y = "search_time", data = df_k, ci=None, scatter_kws={"s": 80})
	plt.savefig('figure/{}_k_search_time_lsh.png'.format(datatype))
	plt.close(0)
	plt.figure(figsize=(15, 15))
	sns.lmplot(x="k", y = "recall", data = df_k, ci=None, scatter_kws={"s": 80})
	plt.savefig('figure/{}_k_recall_lsh.png'.format(datatype))
	plt.close(0)
	plt.figure(figsize=(15, 15))
	sns.lmplot(x="k", y = "touched", data = df_k, ci=None, scatter_kws={"s": 80})
	plt.savefig('figure/{}_k_touched_lsh.png'.format(datatype))
	plt.close(0)
	plt.figure(figsize=(15, 15))
	sns.lmplot(x="k", y = "proportion", data = df_k, ci=None, scatter_kws={"s": 80})
	plt.savefig('figure/{}_k_proportion_lsh.png'.format(datatype))
	plt.close(0)

if __name__ == '__main__':
	# analysis of flann's kdtree in SIFT1M
	tree_num, build_index_time, search_time, recall = get_flann_sift_result()
	box_plot(tree_num, build_index_time, search_time, recall, 'SIFT1M')
	tree_num, build_index_time, search_time, recall = get_flann_gist_result()
	box_plot(tree_num, build_index_time, search_time, recall, 'GIST1M')
	k, L, build_index_time, search_time, recall, touched, proportion = get_lsh_sift_result()
	linear_plot(k, L, build_index_time, search_time, recall, touched, proportion, 'SIFT1M')
	k, L, build_index_time, search_time, recall, touched, proportion = get_lsh_gist_result()
	linear_plot(k, L, build_index_time, search_time, recall, touched, proportion, 'GIST1M')


