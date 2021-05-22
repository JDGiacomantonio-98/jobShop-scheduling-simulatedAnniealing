from math import e as EXP
from os import getcwd
from random import randint, shuffle
from typing import Set, Tuple, List

from scipy.stats import uniform


# TO-DO: decide who between Problem and SA class should evaluate neighborghs and feasibily 


class Job:
	def __init__(
			self,
			id: int,
			processing_times: Tuple[int],
			release_dates: List[int],
			due_date: int = int(),
			predecessors: Set = None,
			successors: Set = None):

		self.id = id
		self.processing_times = processing_times
		self.releases = release_dates
		self.float: int = 0 # on machine 2
		self.due = due_date
		
		self.predecessors = predecessors if predecessors is not None else set()
		self.successors = successors if successors is not None else set()

		self.weight: int = 1
		self.completion = [self.releases[0] + self.processing_times[0] , self.releases[1] + self.processing_times[1]]

	def get_release(self, onMachine: int = 1):
		return self.releases[onMachine-1]

	def set_release(self, r: int, onMachine: int = 1):
		self.releases[onMachine-1] = r

	def get_processingTime(self, onMachine: int = 1, weighted: bool = False, addFloat: bool = False):
		try:
			if addFloat:
				return self.processing_times[onMachine-1]+self.float if not weighted else ((self.processing_times[onMachine-1] + self.float)/self.weight)
			return self.processing_times[onMachine-1] if not weighted else self.processing_times[onMachine-1]/self.weight
		except IndexError:
			print(f"No such machine for this job: this job is processes on {len(self.processing_times)} machines")
			return None

	def set_float(self, f):
		self.float = f

	def get_completion(self, onMachine: int = 1):
		return self.completion[onMachine-1]

	def set_completion(self, ms: int, onMachine: int = 1):
		self.completion[onMachine-1] = ms

	def add_as_predecessor(self, j):
		self.predecessors.add(j) if j not in self.predecessors else None

	def add_as_successor(self, j):
		self.successors.add(j) if j not in self.successors else None

	def get_predecessors(self):
		return self.predecessors

	def get_successors(self):
		return self.predecessors

	def reset(self):
		self.releases[1] = self.processing_times[0] + self.releases[0]
		self.completion = [self.releases[0] + self.processing_times[0] , self.releases[1] + self.processing_times[1]]

	def to_dict(self) -> dict:
		return self.__dict__

	def to_json(self) -> str:
		json = "{"
		for i, itm in enumerate(self.__dict__.items()):
			json += f"{',' if i != 0 else ''}\n\t'{itm[0]}': {itm[1]}"
		print(json + "\n}")

	def __gt__(self, j) -> bool:
		return True if j in self.predecessors else False

	def __lt__(self, j) -> bool:
		return True if j in self.successors else False
	
	def __repr__(self) -> str:
		return f"<Job 'id':{self.id}, 'release': {self.releases}>"


class Node: 
	"""This class hold the permutation currently under analysis and computes the objective function value it yields"""

	def __init__(self, seq: list, solver):
		self.seq = seq
		self.seq_ideal_onLastMachine: set = None
		self.initial_shift: int  = self.seq[0].get_processingTime() + self.seq[0].get_release()
		self.makespan: int  = None
		self.completion: list = [None, None]

		self.neighborhood = set()
		self.solver = solver

		for i, j in enumerate(self.seq):
			j.reset() # TO-DO: find a clever way to not recompute all this each time we move to a neighborgh
			j.weight = len(self.seq) - i 
		
	def get_seq(self):
		return self.seq[:]
	
	def getNeighborghs(self, rule='fixItems-sptRequest'):
		self.walk_schedule()
		self.seq_ideal_onLastMachine = [j for j in self.solver.problem.sortBy_processingTimes(self.get_seq(), 2, weighted=True, addFloat=True)]

		if rule == "fixItems-sptRequest":
			swaps = {i for i, j in enumerate(self.seq_ideal_onLastMachine) if self.seq[i] != j}
			for i in swaps:
				k = self.seq.index(self.seq_ideal_onLastMachine[i])
				if self.can_swap(i, k):
					self.neighborhood.add(Node.swap(Node(seq=self.get_seq(), solver=self.solver), i, k))
			return self.neighborhood.copy()

		if rule == "soft-rand":
			target = randint(0, len(self.seq)-1)
			to = randint(0, len(self.seq)-1)
			i = 0
			while i <= len(self.seq)//10 and self.can_swap(target, to):
				target = randint(0, len(self.seq)-1)
				to = randint(0, len(self.seq)-1)
				self.neighborhood.add(Node.swap(Node(seq=self.get_seq(), solver=self.solver), target, to))
				i += 1
				continue
			return self.neighborhood.copy()

	def apply_perturbation(self, level="soft"):
		i: int  = 0
		target = randint(0, len(self.seq)-1)
		to = randint(0, len(self.seq)-1)
		if level == 'soft':
			while i <= len(self.seq)//10 and self.can_swap(target, to):
				target = randint(0, len(self.seq)-1)
				to = randint(0, len(self.seq)-1)
				Node.swap(self, target, to)
				i += 1
				continue

	def can_swap(self, target: int, to: int):
		return True if self.seq[target].get_release() >= self.seq[to-1].get_completion() else False

	@staticmethod
	def swap(n, target: int, to: int):
		n.seq[target], n.seq[to] = n.seq[to], n.seq[target]
		return n

	def eval(self):
		return self.walk_schedule()[1] if self.completion[0] is None else self.completion[1]

	def get_completion(self, onMachine: int = 1):
		return 0 if self.completion[onMachine-1] is None else self.completion[onMachine-1]

	def walk_schedule(self, returnList: bool = True):
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

		self.completion[1] = self.seq[0].get_completion(2)
		self.makespan = self.seq[0].get_completion(2)

		for i in range(1, len(self.seq)):
			self.seq[i].set_completion(max(self.seq[i-1].get_completion(1), self.seq[i].get_release(1)) + self.seq[i].get_processingTime(1), onMachine=1)
			self.seq[i].set_release(self.seq[i].get_completion(1), onMachine=2)
			self.seq[i].set_completion(max(self.seq[i-1].get_completion(2), self.seq[i].get_release(2)) + self.seq[i].get_processingTime(2), onMachine=2)
			# self.seq[i].set_release(max(self.seq[i-1].get_completion(2), self.seq[i].get_completion(1)), onMachine=2)
			self.completion[1] += self.seq[i].get_completion(onMachine=2)
			self.seq[i-1].set_float(max(0, self.seq[i].get_completion(1)-self.seq[i-1].get_completion(2)))
		self.makespan = self.seq[i].get_completion(onMachine=2)

		if self.completion[0] is None:
			self.completion[0] = 0
			for j in self.seq:
				self.completion[0] += j.get_completion(onMachine=1)

		if returnList:
			return self.completion[:]
		return self.completion[1]


class SimulatedAnnieling:
	def __init__(self, p, head: Node=None, T0: int=15000, coolingFunc=None, heatingFunc=None, probaFunc=None):
		self.problem = p
		self.head = self.set_initial_sol(rule="erd-spt") if head is None else head    #  the current node to be evalauted (a feasible permutation of jobs) to be evaluated
		self.sol_bounds = [self.head.get_completion(0), self.head.get_completion(1)]
		self.T = T0
		self.coolingProfile = self.linearCooling if coolingFunc is None else coolingFunc
		self.heatingProfile =  self.heater if coolingFunc is None else coolingFunc
		self.probaEngine = sigmoidFunc if probaFunc is None else probaFunc


	def set_initial_sol(self, rule: str = "erd-spt"):
		if rule == "rand":
			seq = list(self.problem.jobs)
			shuffle(seq)

			return Node(seq=seq, solver=self)

		if rule == "r-max":
			seq = self.problem.sortBy_release()
			ss = seq.pop(len(seq)-1)
			seq.insert(0, ss)
			seq = self.problem.sortBy_processingTimes(jobs=seq[1:])
			seq.insert(0, ss)

			return Node(seq=seq, solver=self)
		
		if rule == "erd-spt":
			n = Node(self.problem.sortBy_release(), solver=self)
			n.walk_schedule()
			s2 = self.problem.sortBy_processingTimes(n.get_seq(), 2, weighted=True, addFloat=True)
			fix = {i for i in range(len(s2)) if n.seq[i] == s2[i]}
			print(f"matching positions: {len(fix)}") # TO-DO: further investigation of this to merge beeft of the two

			return n # ERD


		return list() # TO-DO: extend methods to find better initial solutions

	def set_head(self, n: Node):
		self.head = n

	def set_T(self):
		self.T = self.coolingProfile(self.T)
	
	def linearCooling(self, cooling_rate: float = 0.90):
		self.T *= cooling_rate
		return self.T
	
	def linearHeating(self, heating_rate: float = 0.90):
		self.T *= heating_rate
		return self.T


	def eval_move(self, to: Node, temperature: float):
		if uniform.rvs() > self.probaEngine(self.head, neighborgh=to, temperature=temperature):
			self.head = to
			print(f"head moved! C MAX: {self.head.eval()}")
	
	def solve(self, timeLimit=60, firstImprovement: bool = False, bestImprovement: bool = True):
		if firstImprovement:
			while True:
				for n in self.head.getNeighborghs(rule="fixItems-sptRequest"):
					if n.eval() < self.head.eval():
						self.eval_move(n)
						break
					continue

		if bestImprovement:
			i=0
			while True:
				best = self.head
				for n in self.head.getNeighborghs(rule="fixItems-sptRequest"):
					best = n if n.eval() < best.eval() else best
				i += 1 if best == self.head else 0
				if i > 3: # algorithm converged into local minima
					i=0
					print("increasing temperature ...")
					self.T = 15000
					print("applying perturbation ...")
					self.head.apply_perturbation("soft")
					continue
				self.eval_move(best, temperature=self.coolingProfile(self.T))


class Problem:
	def __init__(self, id: int=1, solver = None, jobs_file_uri: str = f"{getcwd()}\\instances\\test_small.txt"):
		
		self.id = id
		self.machines: int = 2
		self.jobs: Tuple[Job] = tuple()
		self.job = self.load_jobs_from_file(jobs_file_uri)

		self.objective = None # dict of possible objective funcs
		self.solver = SimulatedAnnieling(p=self) if solver is None else solver


	def load_jobs_from_file(self, file_uri: str, returnList: bool = False):

		with open(file_uri, "r") as f:
			for line in f.readlines():
				line = line[:-1].split(" ")
				self.jobs += (Job(id=str(int(line[0])+1), processing_times=(int(line[1]), int(line[2])), release_dates=[int(line[3]), int(line[3])+int(line[1])]),)

		if returnList:
			return list(self.jobs)
	
	def sortBy_release(self, jobs: list = None, onMachine: int = 1, custom: bool = False, append_stats: bool = False):

		seq = list(self.jobs) if jobs is None else jobs
		
		if custom:
			for i in range(0, len(self.jobs)):
				head = seq[i].get_release()
				for j in range(i+1, len(self.jobs)):
					if head >= seq[j].get_release():
						i+=1 # head moves
						x=seq.pop(j)
						seq.insert(i-1, x) # outlier is swapped right before the head
			# TO-TO: find how to write ordering problem in a bisectional fashion
		else:
			seq.sort(key=lambda x:x.get_release(onMachine=onMachine))
			if append_stats:
				return (seq, min(seq), max(seq))
			return seq

	def sortBy_processingTimes(self, jobs: list = None, onMachine: int = 1, weighted: bool = False, addFloat: bool = False, custom: bool = False):

		seq = list(self.jobs) if jobs is None else jobs

		if custom:
			for i in range(0, len(self.jobs)):
				head = seq[i].get_release()
				for j in range(i+1, len(self.jobs)):
					if head >= seq[j].get_release():
						i+=1 # head moves
						x=seq.pop(j)
						seq.insert(i-1, x) # outlier is swapped right before the head
			# TO-TO: find how to write ordering problem in a bisectional fashion
		else:
			seq.sort(key=lambda x:x.get_processingTime(onMachine=onMachine, weighted=weighted, addFloat=addFloat))
			return seq


def sigmoidFunc(head: Node, neighborgh: Node, temperature: int, minimisation=True) -> float:
	return 1/(1+EXP**((head.eval()-neighborgh.eval())/temperature)) if minimisation else 1/(1+EXP**(-(head.eval()-neighborgh.eval())/temperature))


"""
p = Problem()
print(p.solver.head.eval())

for i in range(0, 20):
	p = Problem(jobs_file_uri=f"{getcwd()}\\instances\\test100_{i}.txt")
	print(p.solver.head.eval())
"""

for i in range(0, 20):
	p = Problem(jobs_file_uri=f"{getcwd()}\\instances\\test100_{i}.txt")
	p.solver.solve(timeLimit=60)