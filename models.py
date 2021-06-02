from copy import copy
from math import ceil
from math import e as EXP
from os import getcwd, scandir
from random import choice, randint, shuffle
from secrets import token_hex
from time import perf_counter, time
from typing import Dict, List, Set, Tuple

from scipy.stats import uniform


class Job:
	def __init__(
			self,
			id: int,
			processing_times: Tuple[int],
			release_dates: List[int]):

		self.id = id
		self.processing_times = processing_times

		# the following dicts will store jobs data for each new node explored and will drop data if the node is discarded. Keys point to node IDs
		# Initial solution node has ID=0
		self.releases: Dict[str, List[int]] = {
			"init": release_dates,
		}

		self.completion: Dict[str, List[int]] = {
			"init": [self.releases["init"][0] + self.processing_times[0], self.releases["init"][1] + self.processing_times[1]],
		}

		self.weight: Dict[str, int] = {
			"init": 1,
		}

	def __eq__(self, j: object) -> bool:
		return True if self.id == j.id else False
	
	def __ne__(self, j: object) -> bool:
		return True if self.id != j.id else False

	def __repr__(self) -> str:
		return f"J{self.id}"


	def add_node(self, node, parent): 
		self.completion[node.id] =  [None for _ in range(node.solver.problem.machines)] if parent is None else copy(self.completion[parent.id])
		if node.is_initialSol:
			self.releases[node.id] = copy(self.releases["init"])

	def getProcessingTime(self, onMachine: int = 1, weighted: bool = False, node=None):
		try:
			return self.processing_times[onMachine-1] if not weighted else self.processing_times[onMachine-1]/self.weight["init" if node is None else node.id]
		except IndexError:
			print(f"No such machine for this job: this job is processes on {len(self.processing_times)} machines")
			return None

	def getRelease(self, node =  None, onMachine: int = 1):
		return self.releases["init" if node is None else node.id][onMachine-1]

	def setRelease(self, r: int, node: int, onMachine: int = 1):
		try:
			self.releases[node.id][onMachine-1] = r
		except KeyError:
			self.releases[node.id] = [r]
		except IndexError:
			self.releases[node.id].append(r)

	def getCompletion(self, node = None, onMachine: int = 1):
		return self.completion["init" if node is None else node.id][onMachine-1]

	def setCompletion(self, c: int, node: int, onMachine: int = 1):
		self.completion[node.id][onMachine-1] = c
	

class Node: 
	"""This class hold the permutation currently under analysis and computes the objective function value it yields"""

	def __init__(self, seq: list, solver, id: int = None, is_initial: bool = False, childOf = None):
		self.solver = solver
		self.id = self.solver.nodes_count + 1 if id is None else 0 if is_initial else id
		self.is_initialSol: bool = is_initial
		self.seq = seq
		self.seq_ideal_onLastMachine: List = None
		self.fixed: Set = None
		self.swaps: Set = None 
		self.completion: List[int, int] = [None, -1] # [function value, schedule index since it has been walked throught] 

		self.parent = None if childOf is None else childOf
		self.neighborhood: Set = set()
		self.probaMove: float = None

		self.solver.nodes_count += 1

		for j in self.seq:
			j.add_node(self, self.parent)
		

	def get_seq(self):
		return self.seq[:]

	def eval(self):
		return self.completion[0] if self.completion[1] == len(self.seq)-1 else self.walk_schedule()
	
	def getCompletion(self, onMachine: int = 2, up_to: int = None): # tries to access job completion for the node, on error trigger a schedule walk up to that job
		try:
			return self.seq[up_to].getCompletion(onMachine=onMachine, node=self)
		except KeyError: # the schedule has not be walked since here
			self.walk_schedule(onMachine=onMachine, up_to=up_to)

	def walk_schedule(self, onMachine: int = None, up_to: int = None):
		"""
		It walks down the sequence of job to compute in linear time schedule parameters

		Objective: MIN{ SUM(C_j) }
		C[m] = ( n * ( K[m] + p[1][m] )) + SUM(from=1, to=n-1){ (n-i) * ( p[i+1][m] + MAX{ r[i+1][m]-MS(i), 0 ) } }

		SUM(C(j)) = n*C[1] + SUM(from=', to=n-1){ (n-i)*C(i)}

		K[1] = r[1][1]
		K[m] = r[1][m-1] + p[1][m-1]

		C[0] = MAX{ 0, r[1][m] }
		C[1] = MS[0] + p[1][m]
		C[i] = p[i][m] + MAX { MS[i-1], r[i][m] }
	
		"""
		# Is there a better way to solve check eval the node?
		ss = perf_counter()
		up_to = len(self.seq)-1 if up_to is None else up_to
		onMachine = self.solver.problem.machines if onMachine is None else onMachine

		if up_to == self.completion[1]:
			return self.completion[0]
		
		if up_to < self.completion[1]: # we already walked up to that
			if up_to >= self.completion[1]//2:
				c = self.completion[0]
				for i in range(self.completion[1], up_to, -1):
					c -= self.seq[i].getCompletion(onMachine=self.solver.problem.machines, node=self)
				return c
			c = 0
			for i in range(up_to+1):
				c += self.seq[i].getCompletion(onMachine=self.solver.problem.machines, node=self)
			return c
			
		# we have to walk farer than done so far, knowing the completion backward already calculated
		start_from = 0 if self.completion[1] < 0 else self.completion[1] + 1
		i = start_from
		for j in self.seq[start_from:up_to+1]:
			if i > up_to:
				break
			if i == 0:
				if j.getCompletion(onMachine=self.solver.problem.machines, node=self) is None: # eval completion of first job on last machine
					j.completion[self.id] = copy(j.completion["init"])
				self.completion[0] = j.getCompletion(onMachine=self.solver.problem.machines, node=self)
				self.completion[1] = 0
				i += 1
				continue
			j.setCompletion(
				max(self.seq[i-1].getCompletion(onMachine=1, node=self), j.getRelease(onMachine=1)) + j.getProcessingTime(1),
				onMachine=1,
				node=self)
			for m in range(2, onMachine+1):
				j.setCompletion(
					max(self.seq[i-1].getCompletion(onMachine=m, node=self), j.getCompletion(onMachine=m-1, node=self)) + j.getProcessingTime(m),
					onMachine=m,
					node=self)
				if m == self.solver.problem.machines:
					self.completion[0] += j.getCompletion(onMachine=m, node=self)
			i += 1
		if onMachine == self.solver.problem.machines:
			self.completion[1] = up_to # remember the farest walked index of the schedule

		print(f"Node {self.id} (Child of Node {self.parent.id if self.parent is not None else 'None'}) : Eval schedule in: {perf_counter()-ss} -> C:{self.completion[0]}")
		if up_to < len(self.seq)-1:
			return self.seq[up_to].getCompletion(onMachine=onMachine, node=self)
		return self.completion[0]

	def getProbaMove(self):
		if self.probaMove is None:
			self.solver.eval_probaMove(self)
			
		return self.probaMove

	def setProbaMove(self, p: float):
		self.probaMove = p

	def overlap_sequences(self):
		for to in self.eval_swaps(rule="conservative"):
				target = self.seq.index(self.seq_ideal_onLastMachine[to])
				if self.can_swap(target=target, to=to):
					Node.swap(self, target, to)
		return self

	def getNeighborghs(self, rule='explorative'):

		if rule == "complete":
			for x in range(len(self.seq)):
				for y in range(x+1, len(self.seq)):
					yield Node.swap(Node(seq=self.get_seq(), solver=self.solver, childOf=self), y, x, parent=self)

		if rule == "explorative":
			j = self.seq.index(choice(self.seq))
			i = j
			if uniform.rvs() > 0.5: # do swaps upstream, downstream otherwise
				while i < len(self.seq):
					yield Node.swap(Node(seq=self.get_seq(), solver=self.solver, childOf=self), j, i)
					i += 1
			else:
				while i >= 0:
					yield Node.swap(Node(seq=self.get_seq(), solver=self.solver, childOf=self), j, i)
					i -= 1

		if rule == "pullDownstream-alignedfix":
			self.eval_ideal_downstream()
			for to in self.eval_swaps(rule="conservative"):
				target = self.seq.index(self.seq_ideal_onLastMachine[to])
				i: int = 1
				if self.can_swap(target, to):
					n = Node(seq=self.get_seq(), solver=self.solver, id = i)
					self.solver.eval_probaMove(n, temperature=self.solver.annielingProcess())
					yield Node.swap(n, to, target)
					i += 1
			return self.neighborhood

		if rule == "rand-alignedfix":
			self.eval_ideal_downstream()
			for to in self.eval_swaps(rule="conservative"):
				for target in self.get_swaps() - {to}:
					if self.can_swap(target, to):
						n = Node(seq=self.get_seq(), solver=self.solver)
						self.solver.eval_probaMove(Node.swap(n, to, target), temperature=self.solver.annielingProcess())
						yield n
						self.neighborhood.add(n)

		if rule == "rand":
			target = randint(0, len(self.seq)-1)
			to = randint(0, len(self.seq)-1)
			while self.can_swap(target, to):
				target = randint(0, len(self.seq)-1)
				to = randint(0, len(self.seq)-1)
				yield Node.swap(Node(seq=self.get_seq(), solver=self.solver), to, target)
		
		if rule == "stress-best":
			pass

	def apply_perturbation(self, level: str = "soft-rand-alignedfix"):
		i: int  = 0
		self.solver.sol_bounds[1] = self.eval()

		if level == 'soft-rand-alignedfix':
			to: int = choice(tuple(self.eval_swaps(rule="conservative")))
			target = self.seq.index(self.seq_ideal_onLastMachine[to])
			while True:
				if self.can_swap(target, to):
					Node.swap(self, target, to)
					i += 1
				if i <= ceil(len(self.seq)/10):
					try:
						to: int = choice(tuple(self.eval_swaps(rule="conservative")))
						target = self.seq.index(self.seq_ideal_onLastMachine[to])
					except Exception:
						print("debug")
				else:
					break
				continue
	
	def eval_swaps(self, rule="conservative", returnObj: bool = True):
		if rule == 'conservative':
			# TO-DO: this seems not to work
			self.swaps = {i for i, j in enumerate(self.seq_ideal_onLastMachine) if self.seq[i] != j}

		if returnObj:
			return self.swaps.copy()
	
	def get_swaps(self):
		return self.swaps.copy()

	def eval_fixed(self):
		self.fixed = {i for i, j in enumerate(self.seq_ideal_onLastMachine) if self.seq[i] == j} if self.fixed is None else self.fixed

	def can_swap(self, target: int, to: int):
		if target > to:
			return True if self.seq[target].getRelease(onMachine=1) <= self.seq[to-1].getCompletion(node=self, onMachine=1) else False
		return True if self.seq[to].getRelease(onMachine=1) <= self.seq[target-1].getCompletion(node=self, onMachine=1) else False

	@staticmethod
	def swap(n, target: int, to: int, parent=None):
		if to == 0:
			n.seq[target].completion[n.id] = n.seq[target].completion["init"]
		if  to != 0 and parent is not None:
			n.completion[0] = parent.walk_schedule(up_to=to-1) 
			n.completion[1] = to-1
		n.seq[target], n.seq[to] = n.seq[to], n.seq[target]
		return n

	def eval_ideal_downstream(self):
		self.walk_schedule(returnList=False) if self.completion[0] is None else None
		self.seq_ideal_onLastMachine = [j for j in self.solver.problem.sortBy_processingTimes(self.get_seq(), 2, weighted=True)] if self.seq_ideal_onLastMachine is None else self.seq_ideal_onLastMachine
		self.eval_fixed()


class SimulatedAnnieling:
	def __init__(self, p, head: Node=None, T0: int=15000, anniealingFunc=None, probaFunc=None):
		self.id = p.id # TO-DO: this will address the possibility to link multiple solvers to same problem instance
		self.problem = p
		self.open_nodes: Set = set()
		self.nodes_count: int = 0
		self.best_found: str = [str(), int()] # stores the best schedule and the yielded problem value
		self.head = self.set_initial_sol(rule="compressFloats-sptDownstream") if head is None else head   #  the current node to be evalauted (a feasible permutation of jobs) to be evaluated ... this method has to e changed with the best on averange initial sols generator

		self.T = T0
		self.annielingProcess = self.linearCooling if anniealingFunc is None else anniealingFunc
		self.probaEngine = self.sigmoidProbaFunc if probaFunc is None else probaFunc


	def set_initial_sol(self, rule: str = "compressFloats-sptDownstream"):
		
		if rule == "compressFloats-sptDownstream":
			#ss = perf_counter()
			n = Node(self.problem.sortBy_release(), solver=self, is_initial=True)
			head: int = n.seq[0].getRelease(node=n)
			horizon: int = n.seq[0].getProcessingTime()
			k: int = 0 # index of last job allocated in the sequence
			target: int = int()
			while k < len(n.seq)-1: # repeat until there are jobs to place
				for i in range(k, len(n.seq)): # compute max time excursion where decisional trade-offs exists
					if n.seq[i].getRelease(node=n) != head:
						break
					if n.seq[i].getProcessingTime(1) >= horizon:
						horizon = n.seq[i].getProcessingTime(1)
						target = i
				horizon += head

				min:int = horizon + n.seq[target].getProcessingTime(2)
				for i in range(k, len(n.seq)): # select among the horizon-included jobs the one that minimize the float behind it
					if n.seq[i].getRelease(node=n) + n.seq[i].getProcessingTime(1) > horizon:
						break
					if n.seq[i].getRelease(node=n) + n.seq[i].getProcessingTime(1) + n.seq[i].getProcessingTime(2) < min:
						min = n.seq[i].getRelease(node=n) + n.seq[i].getProcessingTime(1) + n.seq[i].getProcessingTime(2)
						target = i

				target = n.seq.pop(target)
				n.seq.insert(k, target) # insert the job on the front of the seq and let the other shift by one
				head = n.walk_schedule(onMachine=1, up_to=k)
				for i in range(k+1, len(n.seq)): # updates the release dates among the horizon-included jobs considering that target will be placed before them
					if n.seq[i].getRelease(node=n) + n.seq[i].getProcessingTime(1) > horizon:
						break
					n.seq[i].setRelease(max(head, n.seq[i].getRelease(node=n)), node=n) # evaluate the node schedule completion in a lazy-fashion : up to the indicated node (node 0)
				head = n.seq[k+1].getRelease(node=n)
				horizon: int = n.seq[k+1].getProcessingTime(1)
				k += 1
				'''
				nn = Node(self.problem.sortBy_processingTimes(jobs=n.get_seq()[k+1:]), solver=self, id=-1) # arrange downstream-to-target jobs accordingly to SPT on M2
				if nn.walk_schedule(onMachine=1) + min <= min + n.seq[k].getProcessingTime(onMachine=2): # compute only completion on M1 and compare it with amx completion on M2
					n = Node(n.seq[:k+1].extend(nn.get_seq()))
					return n
				else: # prepare remaining jobs to next iteration
					head = n.walk_schedule(onMachine=1, up_to=k)
					for i in range(k, len(n.seq)): # updates the release dates among the horizon-included jobs considering that target will be placed before them
						if n.seq[i].getRelease() + n.seq[i].getProcessingTime() > horizon:
							break
						n.seq[i].setRelease(head, node=n.id) # evaluate the node schedule completion in a lazy-fashion : up to the indicated node (node 0)
					k += 1
				'''
			#print(f"Initial solution generated in: {perf_counter()-ss}")
			self.save_as_best(n)

			return n

		if rule == "erd-spt": # deprecated probably
			n = Node(seq=self.problem.sortBy_release(), solver=self)
			n.eval_ideal_downstream()
			print(f"matching positions: {len(n.fixed)}") # TO-DO: further investigation of this to merge beeft of the two

			return n.overlap_sequences() # ERD

		if rule == "rand":
			seq = list(self.problem.jobs)
			shuffle(seq)

			return Node(seq=seq, solver=self, is_head=True)

		if rule == "rmax":
			seq = self.problem.sortBy_release()
			ss = seq.pop(len(seq)-1)
			seq.insert(0, ss)
			seq = self.problem.sortBy_processingTimes(jobs=seq[1:])
			seq.insert(0, ss)

			return Node(seq=seq, solver=self, is_head=True)
		
	def compare_initial_sols(self):
		for rule in ["rand", "rmax", "erd-spt", "compressFloats-sptDownstream"]:
			n = self.set_initial_sol(rule=rule)
			print(f'"{rule}" rule ->  C : {n.eval()}')
		return n
	
	def linearCooling(self, rate: float = 0.90):
		if uniform.rvs() > 0.9990: # apply perturbation in temperature
			self.T = 10000*max(1.5, 3*uniform.rvs())
		else:
			self.T *= rate

	def eval_probaMove(self, to: Node, temperature: float=None):
		temperature = self.T if temperature is None else temperature
		to.setProbaMove(self.probaEngine(neighborgh=to, temperature=temperature) if self.probaEngine == self.sigmoidProbaFunc else self.probaEngine(self.head, neighborgh=to, temperature=temperature))

	def sigmoidProbaFunc(self, neighborgh: Node, temperature: int, minimisation=True) -> float:
		k = ((neighborgh.eval() - self.head.eval())/temperature) if minimisation else ((self.head.eval() - neighborgh.eval())/temperature)
		if k <= 230:
			return 1/(1+(EXP**k))
		return 0

	def eval_move(self, to: Node):
		return True if self.head != to and uniform.rvs() <= to.getProbaMove() else False
	
	def move_head(self, to: Node):
		self.head = to
		#print(f"head moved! C: {self.head.eval()}")
			
	def solve(self, timeLimit=60, rule: str = "probabilistic"):
		print(f"\nInitial head set! C: {self.head.eval()}")

		if rule == "probabilistic": # SA explore also bad moves probabilistically !
			while perf_counter()-self.problem.starting_time <= timeLimit:
				#ss = perf_counter()
				progress = (perf_counter()-self.problem.starting_time)/timeLimit
				for n in self.head.getNeighborghs(rule="complete" if progress <= 0.60 else "explorative"):
					self.annielingProcess(rate=1+(self.problem.starting_time/(max(0.96, self.T*uniform.rvs())*perf_counter())))
					#print(f"T : {self.T}")
					if self.eval_move(n):
						self.open_nodes.add(n)
					if n.eval() < self.best_found[1]:
						self.save_as_best(n)
				try:
					self.move_head(choice(tuple(self.open_nodes)))
				except IndexError:
					try:
						for n in self.head.getNeighborghs(rule="rand-alignedfix"):
							if self.eval_move(n):
								self.open_nodes.add(n)
						self.move_head(choice(tuple(self.open_nodes)))
						self.open_nodes - {self.head}
					except IndexError:
						self.print_answer()
						self.save_stats(useBin=self.problem.stats_bin is not None)
						quit()
				else:
					self.open_nodes -= {self.head}
					self.annielingProcess(0.96)
				#print(f"explore neighborhood in: {perf_counter()-ss}")
			self.print_answer()
			self.save_stats(useBin=self.problem.stats_bin is not None)
			
		if rule == "firstImprovement":
			while perf_counter()-self.problem.starting_time <= timeLimit:
				for n in self.head.getNeighborghs(rule="pullDownstream-alignedfix"):
					if n.eval() < self.head.eval():
						self.eval_move(n)
						break
					continue
			self.print_answer()
			self.save_stats()

		if rule == "bestImprovement": # hill-climbing
			i=0	
			while perf_counter()-self.problem.starting_time <= timeLimit:
				best = self.head
				for n in self.head.getNeighborghs(rule="pullDownstream-alignedfix"):
					best = n if n.eval() < best.eval() else best # TO-DO: is it right to remove bad nodes?
				i += 1 if best == self.head else 0
				if i > 3: # algorithm converged into local minima
					i=0
					#print("increasing temperature ...")
					self.T = 15000
					#print("applying perturbation ...")
					self.head.apply_perturbation("soft-rand-alignedfix")
					continue
				self.move_head(best) if self.eval_move(best) else None
			self.print_answer()
			self.save_stats()

	def benchmark_singleInstance(self, runs: int = 20):
		for _ in range(runs):
			self.problem.starting_time = perf_counter()
			self.__init__(self.problem)
			self.solve()

	def save_as_best(self, n: Node):
		self.best_found[0], self.best_found[1] = f"{n.get_seq()}", n.eval()

	def print_answer(self):
		print("\n|********************** || RESULTS || ***********************|")
		print(f"nodes evaluated : {self.nodes_count}")
		print(f"Cmin found = {self.best_found[1]}")
		print(f"best schedule :\n{self.best_found[0]}")
	
	def save_stats(self, folder_uri: str = f"{getcwd()}\\stats", useBin: bool = False):
		stats: str = f"** {self.problem.dataset_loc} **\nBEST C : {self.head.eval()}\nBest sequence : \n{self.head.get_seq()}\n"
		if useBin:
			self.problem.stats_bin[0].append(stats)
		else:
			with open(f"{folder_uri}\\run-instance-{self.problem.id}_{self.id}", "w") as f:
				f.write(stats)
	

class Problem:
	def __init__(self, id: int = token_hex(8), solver = None, jobs_file_uri: str = f"{getcwd()}\\instances\\test_small.txt", stats_bin: list = None):
		
		self.id = id
		self.machines: int = 2
		self.jobs: Tuple[Job] = tuple()
		self.dataset_loc: str = jobs_file_uri
		self.load_jobs_from_file(jobs_file_uri)
		self.stats_bin = stats_bin

		self.solver = SimulatedAnnieling(p=self) if solver is None else solver
		self.starting_time = perf_counter()


	def load_jobs_from_file(self, file_uri: str, returnList: bool = False):

		with open(file_uri, "r") as f:
			for line in f.readlines():
				line = line[:-1].split(" ")
				self.jobs += (Job(id=str(int(line[0])+1), processing_times=(int(line[1]), int(line[2])), release_dates=[int(line[3]), int(line[3])+int(line[1])]),)

		if returnList:
			return list(self.jobs)
	
	def sortBy_release(self, jobs: list = None, onMachine: int = 1, append_stats: bool = False):

		seq = list(self.jobs) if jobs is None else jobs
		
		seq.sort(key=lambda x:x.getRelease(onMachine=onMachine))
		if append_stats:
			return (seq, min(seq), max(seq))
		return seq

	def sortBy_processingTimes(self, jobs: list = None, onMachine: int = 1, weighted: bool = False, caller: Node = None):

		seq = list(self.jobs) if jobs is None else jobs

		if caller is None:
			weighted = False
		seq.sort(key=lambda x:x.getProcessingTime(onMachine=onMachine, weighted=weighted, node=caller))
		return seq
		

def assign_id_to_jobs_in_file(source_file_uri: str = f"{getcwd()}\\instances\\test_all.txt", target_file_uri: str = f"{getcwd()}\\instances\\all_merged.txt"):
	with open(source_file_uri, "r") as source:
		with open(target_file_uri, "w") as target:
			i = 0
			for line in source.readlines():
				line = line.split(" ")
				line[0] = str(i)
				target.write(line[0] + " " + line[1] + " " + line[2] + " " + line[3])
				i+=1

def throw_stats_to_file(stats_bin: list, file_loc_uri: str = f"{getcwd()}\\stats"):
	with open(f"{file_loc_uri}\\run-instance-{stats_bin[1]}", "w") as f:
		for i in stats_bin[0]:
			f.write(i)
'''
for f in scandir(f"{getcwd()}\\instances"):
	print(f"\n** INSTANCE : {f.name} **")
	p = Problem(jobs_file_uri=f"{getcwd()}\\instances\\{f.name}")
	#p.solver.compare_initial_sols()
	p.solver.benchmark_singleInstance()
'''


p = Problem(jobs_file_uri=f"{getcwd()}\\instances\\test100_2.txt")
p.solver.solve()

'''
stats_bin = [[], int(time())]
for f in scandir(f"{getcwd()}\\instances"):
	print("trying to solve the next problem  ...")
	p = Problem(jobs_file_uri=f"{getcwd()}\\instances\\{f.name}", stats_bin=stats_bin)
	p.solver.solve(timeLimit=60)
throw_stats_to_file(stats_bin)
'''