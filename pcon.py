import time
import sys
from random import choice, randint
import math
import collections
from binary_heap import *
from scipy.sparse import csr_matrix
import scipy as sp
import scipy.sparse.linalg as splinalg
import numpy as np




class Graph:

	def __init__(self, dataset):
		self.dataset = dataset
		self.adj_list, self.deg = self.graph_list()
		self.n=len(self.adj_list)
		self.m = 0
		for u in self.adj_list:
			self.m += len(self.adj_list[u])
		self.m = self.m / 2
		print("number of nodes" + str(self.n))
		print("number of edges" + str(self.m))
		


	def graph_list(self):
		starttime = time.time()
		adj_list = {}
		print("........................")
		print(self.dataset + " is loading...")
		file=open(self.dataset)
		while 1:
			lines = file.readlines(50000)
			if not lines:
				break
			for line in lines:
				line=line.split()
				f_id,t_id=int(line[0]),int(line[1])
				if f_id in adj_list:
					adj_list[f_id].add(t_id)
				else:
					adj_list[f_id] = {t_id}
		file.close()
		deg = {}
		for u in adj_list:
			deg[u] = len(adj_list[u])
		endtime=time.time()
		print("load_time(s)"+str(endtime-starttime))
		return adj_list, deg


	def compu_conductance(self, S):
		d = {}
		dout = {}
		sum_d = 0
		sum_dout = 0
		for u in S:
			d[u] = len(self.adj_list[u])
			sum_d += d[u]
			dout[u] = d[u] - len(set(self.adj_list[u]) & S)
			sum_dout += dout[u]
		condu = sum_dout / min(sum_d, 2 * self.m - sum_d)
		return condu



	def sweep_cut(self,pi):
		starttime = time.time()
		for u in pi:
			pi[u] = pi[u] / self.deg[u]
		pi = sorted(pi.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
		S = set()
		volS = 0
		cutS = 0
		best_condu, best_index, count = 10, 0, 0
		best_set = set()
		for x in pi:
			u = x[0]
			S.add(u)
			count += 1
			cutS = cutS - 2 * len(set(self.adj_list[u]) & S) + self.deg[u]
			volS = volS + self.deg[u]
			if min(volS, 2 * self.m - volS)!=0 and cutS / min(volS, 2 * self.m - volS) < best_condu:
				best_condu = cutS / min(volS, 2 * self.m - volS)
				best_index = count
		for x in range(best_index):
			best_set.add(pi[x][0])
		if len(best_set) > self.n/ 2:
			best_set = set(self.adj_list) - set(best_set)
		endtime = time.time()
		sweep_time = endtime - starttime
		return sweep_time, best_set,best_condu




	def Pcon_de(self):
		starttime = time.time()
		S = set(self.adj_list)
		dr={}
		mymin_heap = MinHeap([])
		for u in self.adj_list:
			dr[u]=1
			mymin_heap.insert([u, 1])
		best_g = 0
		D=[]
		best_index=0
		count=0
		g_s_upper = 2 * self.m
		g_s_lower=4*self.m
		volS=2*self.m
		while S:
			u = mymin_heap.remove()[0]
			volS-=self.deg[u]
			S.remove(u)
			D.append(u)
			count+=1
			g_s_upper=g_s_upper-2*len(set(self.adj_list[u])&S)
			g_s_lower=g_s_lower-2*self.deg[u]
			if g_s_lower!=0 and g_s_upper/g_s_lower>best_g and volS<=self.m:
				best_index=count
				best_g=g_s_upper/g_s_lower
			for v in self.adj_list[u]:
				if v in S:
					dr[v]=dr[v]-1/self.deg[v]
					mymin_heap.decrease_key(v, dr[v])

		remove_set=set()
		for x in range(best_index):
			remove_set.add(D[x])
		best_set = set(self.adj_list) - remove_set
		endtime = time.time()
		best_condu=self.compu_conductance(best_set)
		peeling_time = endtime - starttime
		return peeling_time, best_condu


	def Pcon_Core(self):
		starttime = time.time()
		d ={}
		mymin_heap = MinHeap([])
		for u in self.adj_list:
			d[u]=len(self.adj_list[u])
			mymin_heap.insert([u, d[u]])
		temp = set(self.adj_list)
		D=[]
		while temp:
			u = mymin_heap.remove()[0]
			temp.remove(u)
			D.append(u)
			for v in self.adj_list[u]:
				if v in temp:
					d[v] = d[v] - 1
					mymin_heap.decrease_key(v, d[v])
		pi=[]
		for i in range(self.n-1,-1,-1):
			pi.append(D[i])
		S = set()
		volS = 0
		cutS = 0
		best_condu, best_index, count = 10, 0, 0
		best_set = set()
		for x in pi:
			u = x
			S.add(u)
			count += 1
			cutS = cutS - 2 * len(set(self.adj_list[u]) & S) + self.deg[u]
			volS = volS + self.deg[u]
			if min(volS, 2 * self.m - volS)!=0 and cutS / min(volS, 2 * self.m - volS) < best_condu:
				best_condu = cutS / min(volS, 2 * self.m - volS)
				best_index = count
		for x in range(best_index):
			best_set.add(pi[x])
		if len(best_set) > self.n/ 2:
			best_set = set(self.adj_list) - set(best_set)
		endtime = time.time()
		peeling_time = endtime - starttime
		return peeling_time, best_condu




	def SC(self):
		starttime=time.time()
		data=[1 for i in range(int(2*self.m))]
		indptr=[0]
		pre_sum=0
		indices=[]
		gd = []
		for i in range(self.n):
			gd.append(1/math.sqrt(self.deg[i]))
			pre_sum+=self.deg[i]
			indptr.append(pre_sum)
			for node in self.adj_list[i]:
				indices.append(node)

		A=csr_matrix((data, indices, indptr), shape=(self.n, self.n))
		D_sqrt_neg = sp.sparse.spdiags(gd, 0, self.n, self.n)
		L = sp.sparse.identity(self.n) - D_sqrt_neg.dot((A.dot(D_sqrt_neg)))

		emb_eig_val, p = splinalg.eigsh(L, which='SM', k=2)
		pi = np.real(p[:, 1])
		pi=np.argsort(pi)

		S = set()
		volS = 0
		cutS = 0
		best_condu, best_index, count = 10, 0, 0
		best_set = set()
		for x in pi:
			u = x
			S.add(u)
			count += 1
			cutS = cutS - 2 * len(set(self.adj_list[u]) & S) + self.deg[u]
			volS = volS + self.deg[u]
			if min(volS, 2 * self.m - volS)!=0 and cutS / min(volS, 2 * self.m - volS) < best_condu:
				best_condu = cutS / min(volS, 2 * self.m - volS)
				best_index = count
		for x in range(best_index):
			best_set.add(pi[x])
		if len(best_set) > self.n/ 2:
			best_set = set(self.adj_list) - set(best_set)
		endtime = time.time()
		SC_time = endtime - starttime
		return SC_time, best_condu


	def ASC(self):
		starttime=time.time()
		x={}
		sum_x=0
		for i in self.adj_list:
			a=randint(0,1)
			if a==0:
				x[i]=-1
				sum_x += x[i]
			else:
				x[i]=1
				sum_x+=x[i]
		inner=sum_x*math.sqrt(self.n)
		x0={}
		for i in self.adj_list:
			x0[i]=x[i]-inner*math.sqrt(self.n)

		for i in range(int(math.log(self.n))):
			new = {}
			for node in self.adj_list:
				new[node]=x0[node]
				for u in self.adj_list[node]:
					new[node]+=x0[u]/(math.sqrt(self.deg[node])*math.sqrt(self.deg[u]))
			for node in self.adj_list:
				x0[node]=new[node]

		pi = sorted(x0.items(), key=lambda kv: (kv[1], kv[0]))
		S = set()
		volS = 0
		cutS = 0
		best_condu, best_index, count = 10, 0, 0
		best_set = set()
		for x in pi:
			u = x[0]
			S.add(u)
			count += 1
			cutS = cutS - 2 * len(set(self.adj_list[u]) & S) + self.deg[u]
			volS = volS + self.deg[u]
			if min(volS, 2 * self.m - volS)!=0 and cutS / min(volS, 2 * self.m - volS) < best_condu:
				best_condu = cutS / min(volS, 2 * self.m - volS)
				best_index = count
		for x in range(best_index):
			best_set.add(pi[x][0])
		if len(best_set) > self.n/ 2:
			best_set = set(self.adj_list) - set(best_set)
		endtime = time.time()
		ASC_time=endtime-starttime
		return ASC_time,best_condu



	def fpush(self, seed,alpha,epsilon):  #NIBBLE_PPR
		starttime = time.time()
		pi, r = {}, {}
		r[seed] = 1
		q = collections.deque()
		q.append(seed)
		while q:
			u = q.popleft()
			for v in self.adj_list[u]:
				if v not in r:
					r[v] = 0
				update=(1 - alpha) * r[u] / self.deg[u]
				r[v] = r[v] + update  # unweighted graph
				if (r[v]- update)/self.deg[v]<epsilon and r[v] / self.deg[v] >= epsilon:
					q.append(v)
			if u not in pi:
				pi[u] = 0
			pi[u] = pi[u] + alpha * r[u]
			r[u] = 0
		endtime = time.time()
		return pi,endtime-starttime

	def hk_relax(self,seed,t,epsilon):
		N=int(2*t*math.log(1/epsilon))+1
		starttime=time.time()
		psis = {}
		psis[N] = 1.
		for i in range(N - 1, -1, -1):
			psis[i] = psis[i + 1] * t / (float(i + 1.)) + 1. #eq(6) of hk-relax
		x = {}  # Store x, r as dictionaries
		r = {}  # initialize residual
		Q = collections.deque()  # initialize queue

		r[(seed, 0)] = 1.
		Q.append((seed, 0))
		while len(Q) > 0:
			(v, j) = Q.popleft()  # v has r[(v,j)] ...
			rvj = r[(v, j)]
			# perform the hk-relax step
			if v not in x: x[v] = 0.
			x[v] += rvj
			r[(v, j)] = 0.
			mass = (t * rvj / (float(j) + 1.)) / self.deg[v]
			for u in self.adj_list[v]:  # for neighbors of v
				next = (u, j + 1)  # in the next block
				if j + 1 == N:
					if u not in x: x[u] = 0.
					x[u] += rvj / self.deg[v]
					continue
				if next not in r: r[next] = 0.
				thresh = math.exp(t) * epsilon * self.deg[u]
				thresh = thresh / (2*N * psis[j + 1])
				if r[next] < thresh and r[next] + mass >= thresh:
					Q.append(next)  # add u to queue
				r[next] = r[next] + mass
		endtime = time.time()
		return x, endtime - starttime



if __name__ == "__main__":
	dataset = sys.argv[1]
	G = Graph(dataset)



	pcon_de_time, best_condu = G.Pcon_de()
	print("pcon_de_time"+str(pcon_de_time))
	print("pcon_de_conductance"+str(best_condu))




	pcon_core_time, best_condu = G.Pcon_Core()
	print("pcon_core_time"+str(pcon_core_time))
	print("pcon_core_conductance"+str(best_condu))




	alpha = 0.01
	t = 10
	epsilon=1/G.m


	avg_push_time=0
	avg_push_condu=0


	avg_HK_time=0
	avg_HK_condu=0

	seed_number=50
	number = 0

	while number < seed_number:
		seed = int(choice(list(G.adj_list)))
		number += 1
		print("seed  " + str(seed))
		pi, fpush_time = G.fpush(seed, alpha, epsilon)
		sweep_time, local_result, best_condu = G.sweep_cut(pi)
		avg_push_time+=fpush_time+sweep_time
		avg_push_condu+=best_condu
		print("push_time" + str(fpush_time+sweep_time))
		print("push_condu" + str(best_condu))

		pi, HK_time = G.hk_relax(seed,t,epsilon)
		sweep_time, local_result, best_condu = G.sweep_cut(pi)
		avg_HK_time+=HK_time+sweep_time
		avg_HK_condu+=best_condu
		print("HK_time" + str(HK_time+sweep_time))
		print("HK_condu" + str(best_condu))

	print("avg_push_time"+str(avg_push_time/seed_number))
	print("avg_push_condu"+str(avg_push_condu/seed_number))

	print("avg_HK_time"+str(avg_HK_time/seed_number))
	print("avg_HK_condu"+str(avg_HK_condu/seed_number))



	ASC_time, best_condu= G.ASC()
	print("ASC_time" + str(ASC_time))
	print("ASC_conductance" + str(best_condu))




	SC_time, best_condu= G.SC()
	print("SC_time" + str(SC_time))
	print("SC_conductance" + str(best_condu))

