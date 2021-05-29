from copy import copy
from math import ceil
from math import e as EXP
from os import getcwd, scandir
from random import choice, randint, shuffle
from secrets import token_hex
from time import perf_counter, time
from typing import Dict, List, Set, Tuple

from scipy.stats import uniform

# TO-DO: decide who between Problem and SA class should evaluate neighborghs and feasibily 


class Job:
	def __init__(
			self,
			id: int,
			processing_times: Tuple[int],
			release_dates: List[int],
			due_date: int = int()):

		self.id = id
		self.processing_times = processing_times
		self.due = due_date

		# the following dicts will store jobs data for each new node explored and will drop data if the node is discarded. Keys point to node IDs
		# Initial solution node has ID=0
		self.releases: Dict[str, List[int]] = {
			"init": release_dates,
		}
		self.float: Dict[str, int] =  {
			"init": 0,
		} # on machine 2 : in this way is like 1||w(j)C(j)

		self.weight: Dict[str, int] = {
			"init": 1,
		}
		self.completion: Dict[str, int] = {
			"init": [self.releases["init"][0] + self.processing_times[0], self.releases["init"][1] + self.processing_times[1]],
		}

	def __eq__(self, j: object) -> bool:
		return True if self.id == j.id else False
	
	def __ne__(self, j: object) -> bool:
		return True if self.id != j.id else False

	def add_node(self, id):
		self.completion[id] = copy(self.completion["init"])
		self.releases[id] = copy(self.releases["init"])
		self.weight[id] = copy(self.weight["init"])
		self.float[id] = copy(self.float["init"])

	def getProcessingTime(self, onMachine: int = 1, weighted: bool = False, addFloat: bool = False, node=None):
		try:
			if addFloat:
				node = "init" if node is None else node
				return ((self.processing_times[onMachine-1] + self.float[node])/self.weight[node]) if not weighted else self.processing_times[onMachine-1] + self.float[node]
			return self.processing_times[onMachine-1] if not weighted else self.processing_times[onMachine-1]/self.weight[node]
		except IndexError:
			print(f"No such machine for this job: this job is processes on {len(self.processing_times)} machines")
			return None

	def getRelease(self, node: int = "init", onMachine: int = 1):
		return self.releases[node][onMachine-1]

	def setRelease(self, r: int, node: int, onMachine: int = 1):
		self.releases[node][onMachine-1] = r

	def set_float(self, f, node: int):
		self.float[node] = f

	def get_float(self, node: int):
		return self.float[node]

	def getCompletion(self, node: int = "init", onMachine: int = 1):
		return self.completion[node][onMachine-1]

	def setCompletion(self, c: int, node: int, onMachine: int = 1):
		self.completion[node][onMachine-1] = c

	def to_dict(self) -> dict:
		return self.__dict__

	def to_json(self) -> str:
		json = "{"
		for i, itm in enumerate(self.__dict__.items()):
			json += f"{',' if i != 0 else ''}\n\t'{itm[0]}': {itm[1]}"
		print(json + "\n}")
	
	def __repr__(self) -> str:
		return f"J{self.id}"
		

class Node: 
	"""This class hold the permutation currently under analysis and computes the objective function value it yields"""

	def __init__(self, seq: list, solver, id: int = 0, is_head: bool = False,):
		self.id = id
		self.is_head: bool = True if self.id == 0 else is_head
		self.seq = seq
		self.seq_ideal_onLastMachine: List = None
		self.fixed: Set = None
		self.swaps: Set = None
		self.initial_shift: int  = self.seq[0].getProcessingTime(1) + self.seq[0].getRelease("init", 1)
		self.makespan: int  = None
		self.completion: list = [None, None]

		self.neighborhood: Set = set()
		self.solver = solver
		self.probaMove: float = None

		for j in self.seq:
			j.add_node(self.id)
		

	def get_seq(self):
		return self.seq[:]
	
	def getCompletion(self, onMachine: int = 1):
		return self.walk_schedule()[onMachine-1] if self.completion[onMachine-1] is None else self.completion[onMachine-1]
	
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

	def getNeighborghs(self, rule='rand'):

		if rule == "pullDownstream-alignedfix":
			self.eval_ideal_downstream()
			for to in self.eval_swaps(rule="conservative"):
				target = self.seq.index(self.seq_ideal_onLastMachine[to])
				i: int = 1
				if self.can_swap(target, to):
					n = Node(seq=self.get_seq(), solver=self.solver, id = i)
					self.solver.eval_probaMove(n, temperature=self.solver.coolingProfile())
					self.neighborhood.add(Node.swap(n, to, target))
					i += 1
			return self.neighborhood

		if rule == "rand-alignedfix":
			self.eval_ideal_downstream()
			i: int = 1
			for to in self.eval_swaps(rule="conservative"):
				for target in self.get_swaps() - {to}:
					if self.can_swap(target, to):
						n = Node(seq=self.get_seq(), solver=self.solver, id = i)
						self.solver.eval_probaMove(Node.swap(n, to, target), temperature=self.solver.coolingProfile())
						self.neighborhood.add(n)
						i += 1
			return self.neighborhood

		if rule == "rand":
			target = randint(0, len(self.seq)-1)
			to = randint(0, len(self.seq)-1)
			i: int = 1
			while self.can_swap(target, to):
				target = randint(0, len(self.seq)-1)
				to = randint(0, len(self.seq)-1)
				self.neighborhood.add(Node.swap(Node(seq=self.get_seq(), solver=self.solver, id = i), to, target))
				i += 1
				continue
			return self.neighborhood
				
		if rule == "explorative": # let this be a python generator
			j = self.seq.index(choice(self.seq))
			i = j
			if uniform.rvs() > 0.5: # do swaps upstream, downstream otherwise
				while i < len(self.seq):
					yield Node.swap(Node(seq=self.get_seq(), solver=self.solver, id = self.id + 1), j, i)
					i += 1
			else:
				while i >= 0:
					yield Node.swap(Node(seq=self.get_seq(), solver=self.solver, id = self.id + 1), j, i)
					i -= 1


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
						print("ciao")
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
			return True if self.seq[target].getRelease(node=self.id, onMachine=1) <= self.seq[to-1].getCompletion(node=self.id, onMachine=1) else False
		return True if self.seq[to].getRelease(node=self.id, onMachine=1) <= self.seq[target-1].getCompletion(node=self.id, onMachine=1) else False

	@staticmethod
	def swap(n, target: int, to: int):
		n.seq[target], n.seq[to] = n.seq[to], n.seq[target]
		return n

	def eval(self):
		return self.walk_schedule()[1] if self.completion[1] is None else self.completion[1]

	def eval_ideal_downstream(self):
		self.walk_schedule(returnList=False) if self.completion[0] is None else None
		self.seq_ideal_onLastMachine = [j for j in self.solver.problem.sortBy_processingTimes(self.get_seq(), 2, weighted=True, addFloat=True)] if self.seq_ideal_onLastMachine is None else self.seq_ideal_onLastMachine
		self.eval_fixed()

	def walk_schedule(self, onMachine: int = 2, up_to: int = None, returnList: bool = True):
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
		# Can we save time when evaluating neighborgs?
		up_to = len(self.seq) if up_to is None else up_to

		if onMachine != 2:
			for i, j in enumerate(self.seq):
				if i > up_to:
					break
				if i == 0:
					self.completion[0] = j.getCompletion(onMachine=1, node=self.id)
					continue
				j.setCompletion(
					max(self.seq[i-1].getCompletion(onMachine=1, node=self.id), j.getRelease(onMachine=1, node=self.id)) + j.getProcessingTime(1),
					onMachine=1,
					node=self.id)
				# self.completion[0] += j.getCompletion(onMachine=1, node=self.id) is it useful to know the sum of completion time on M1? mmh
			return self.seq[i-1].getCompletion(onMachine=1, node=self.id)

		for i, j in enumerate(self.seq):
			if i > up_to:
				break
			if i == 0:
				self.completion[0] = j.getCompletion(onMachine=1, node=self.id)
				self.completion[1] = j.getCompletion(onMachine=2, node=self.id)
				continue
			j.setCompletion(
				max(self.seq[i-1].getCompletion(onMachine=1, node=self.id), j.getRelease(onMachine=1, node=self.id)) + j.getProcessingTime(1),
				onMachine=1,
				node=self.id)
			# j.set_release(j.getCompletion(node=self.id, onMachine=1), onMachine=2, node=self.id) // TO-DO: this instruction add +22% more time, do we need it? 
			j.setCompletion(
				max(self.seq[i-1].getCompletion(onMachine=2, node=self.id, ), j.getCompletion(onMachine=1, node=self.id)) + j.getProcessingTime(2),
				onMachine=2,
				node=self.id)
			self.completion[0] += j.getCompletion(onMachine=1, node=self.id) # TO-DO: do we need this? Or can we save some time?
			self.completion[1] += j.getCompletion(onMachine=2, node=self.id)
			# self.seq[i-1].set_float(max(0, j.getCompletion(onMachine=1, node=self.id) - self.seq[i-1].getCompletion(onMachine=2, node=self.id)), node=self.id)  // TO-DO: this instruction +100% more time, do we need it?
		self.makespan = j.getCompletion(node=self.id, onMachine=2)

		if returnList:
			return self.completion[:]
		return self.completion[1]


class SimulatedAnnieling:
	def __init__(self, p, head: Node=None, T0: int=15000, coolingFunc=None, heatingFunc=None, probaFunc=None):
		self.id = p.id # TO-DO: this will address the possibility to link multiple solvers to same problem instance
		self.problem = p
		self.best_found: str = [str(), int()] # stores the best schedule and the yielded problem value
		self.head = self.set_initial_sol(rule="compressFloats-sptDownstream") if head is None else head   #  the current node to be evalauted (a feasible permutation of jobs) to be evaluated ... this method has to e changed with the best on averange initial sols generator
		self.open_nodes: Set = set()
		self.sol_bounds = [self.head.getCompletion(0), self.head.getCompletion(1)]
		self.T = T0
		self.coolingProfile = self.linearCooling if coolingFunc is None else coolingFunc
		self.heatingProfile =  self.linearHeating if heatingFunc is None else heatingFunc
		self.probaEngine = self.sigmoidProbaFunc if probaFunc is None else probaFunc


	def set_initial_sol(self, rule: str = "erd-spt"):
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
		
		if rule == "erd-spt": # deprecated probably
			n = Node(seq=self.problem.sortBy_release(), solver=self)
			n.eval_ideal_downstream()
			print(f"matching positions: {len(n.fixed)}") # TO-DO: further investigation of this to merge beeft of the two

			return n.overlap_sequences() # ERD

		if rule == "compressFloats-sptDownstream":
			n = Node(self.problem.sortBy_release(), solver=self)
			head: int = n.seq[0].getRelease()
			horizon: int = n.seq[0].getProcessingTime()
			# the following code miss a part of "tracking where we have to insert the new best candidate job" - !! IMPORTANT THIS NEED TO BE SOLVED THIS IS ONLY PSEUDOCODE
			k: int = 0 # index of alst job allocated in teh sequence
			target: int = int()
			while k < len(n.seq)-1: # repeat until there are jobs to place 
				for i in range(k, len(n.seq)): # compute max time excursion where decisional trade-offs exists
					if n.seq[i].getRelease() != head:
						break
					if n.seq[i].getProcessingTime(1) >= horizon:
						horizon = n.seq[i].getProcessingTime(1)
						target = i
				horizon += head

				min:int = horizon + n.seq[target].getProcessingTime(2)
				for i in range(k, len(n.seq)): # select among the horizon-included jobs the one that minimize the float behind it
					if n.seq[i].getRelease() + n.seq[i].getProcessingTime(1) > horizon:
						break
					if n.seq[i].getRelease() + n.seq[i].getProcessingTime(1) + n.seq[i].getProcessingTime(2) < min:
						min = n.seq[i].getRelease() + n.seq[i].getProcessingTime(1) + n.seq[i].getProcessingTime(2)
						target = i

				target = n.seq.pop(target)
				n.seq.insert(k, target) # insert the job on the front of the seq and let the other shift by one
				head = n.walk_schedule(onMachine=1, up_to=k)
				for i in range(k+1, len(n.seq)): # updates the release dates among the horizon-included jobs considering that target will be placed before them
					if n.seq[i].getRelease() + n.seq[i].getProcessingTime(1) > horizon:
						break
					n.seq[i].setRelease(max(head, n.seq[i].getRelease()), node=n.id) # evaluate the node schedule completion in a lazy-fashion : up to the indicated node (node 0)
				head = n.seq[k+1].getRelease()
				horizon: int = n.seq[k+1].getProcessingTime()
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
			self.save_as_best(n)

			return n

	def compare_initial_sols(self):
		rules = ["rand", "rmax", "erd-spt", "compressFloats-sptDownstream"]
		for r in rules:
			n = self.set_initial_sol(rule=r)
			print(f'"{r}" rule ->  C : {n.eval()}')
		return n


	def set_head(self, n: Node):
		n.id = 0
		n.is_head = True
		self.head = n
	
	def linearCooling(self, cooling_rate: float = 0.90):
		self.T *= cooling_rate
		return self.T
	
	def linearHeating(self, heating_rate: float = 0.90):
		self.T *= heating_rate
		return self.T

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
		self.set_head(to)
		# print(f"head moved! C: {self.head.eval()}")
			
	def solve(self, timeLimit=60, rule: str = "probabilistic"):
		print(f"Initial head set! C: {self.head.eval()}")

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

		if rule == "probabilistic": # SA explore also bad moves probabilistically !
			i=0
			while perf_counter()-self.problem.starting_time <= timeLimit:
				best = self.head
				for n in self.head.getNeighborghs(rule="explorative"):
					if self.eval_move(n):
						self.open_nodes.add(n)
					if n.eval() < self.best_found[1]:
						self.save_as_best(n)
				try:
					self.move_head(choice(tuple(self.open_nodes)))
					self.open_nodes - {self.head}
				except IndexError:
					for n in self.head.getNeighborghs(rule="rand-alignedfix"):
						if self.eval_move(n):
							self.open_nodes.add(n)
					self.move_head(choice(tuple(self.open_nodes)))
					self.open_nodes - {self.head}
			self.print_answer()
			self.save_stats(useBin=self.problem.stats_bin is not None)

	def save_as_best(self, n: Node):
		self.best_found[0], self.best_found[1] = f"{n.get_seq()}", n.eval()

	def print_answer(self):
		print(f"{self.best_found[0]} : C = {self.best_found[1]}")
	
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
		
		seq.sort(key=lambda x:x.getRelease(node="init", onMachine=onMachine))
		if append_stats:
			return (seq, min(seq), max(seq))
		return seq

	def sortBy_processingTimes(self, jobs: list = None, onMachine: int = 1, weighted: bool = False, addFloat: bool = False, caller: Node = None):

		seq = list(self.jobs) if jobs is None else jobs

		if caller is None:
			weighted = False
			addFloat = False
		seq.sort(key=lambda x:x.getProcessingTime(onMachine=onMachine, weighted=weighted, addFloat=addFloat, node=caller))
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

for f in scandir(f"{getcwd()}\\instances"):
	print(f"\n** INSTANCE : {f.name} **")
	p = Problem(jobs_file_uri=f"{getcwd()}\\instances\\{f.name}")
	#p.solver.compare_initial_sols()
	p.solver.solve()


'''
stats_bin = [[], int(time())]
for f in scandir(f"{getcwd()}\\instances"):
	print("trying to solve the next problem  ...")
	p = Problem(jobs_file_uri=f"{getcwd()}\\instances\\{f.name}", stats_bin=stats_bin)
	p.solver.solve(timeLimit=60)
throw_stats_to_file(stats_bin)
'''
