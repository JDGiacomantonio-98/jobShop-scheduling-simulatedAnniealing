from math import exp
from os import getcwd
from random import shuffle
from typing import Set, Tuple, List

from scipy.stats import uniform
from scipy.stats.stats import jarque_bera

schedule = Tuple[bool] # the input will be a feasible permutation of n jobs

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
		self.due = due_date
		
		self.predecessors = predecessors if predecessors is not None else set()
		self.successors = successors if successors is not None else set()

		self.current_makespan = [self.releases[0] + self.processing_times[0] , self.releases[1] + self.processing_times[1]]

	def get_release(self, onMachine: int = 1):
		return self.releases[onMachine-1]

	def set_release(self, r: int, onMachine: int = 1):
		self.releases[onMachine-1] = r

	def get_processingTime(self, onMachine: int = 1):
		try:
			return self.processing_times[onMachine-1]
		except IndexError:
			print(f"No such machine for this job: this job is processes on {len(self.processing_times)} machines")
			return None

	def get_makespan(self, onMachine: int = 1):
		return self.current_makespan[onMachine-1]

	def set_makespan(self, ms: int, onMachine: int = 1):
		self.current_makespan[onMachine-1] = ms

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
		self.current_makespan = [self.releases[0] + self.processing_times[0] , self.releases[1] + self.processing_times[1]]

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


class Problem:
	def __init__(self, id: int=1, solver = None, jobs_file_uri: str = f"{getcwd()}\\instances\\test100_0.txt"):
		
		self.id = id
		self.machines: int = 2
		self.jobs: Tuple[Job] = tuple()
		self.job = self.load_jobs_from_file(jobs_file_uri)

		self.objective = None # dict of possible objective funcs
		self.best_sol = 0
		self.solver = SimulatedAnnieling(p=self) if solver is None else solver


	def load_jobs_from_file(self, file_uri: str, returnList: bool = False):

		with open(file_uri, "r") as f:
			for line in f.readlines(): # parse values into Jobs attributes
				line = line[:-1].split(" ")
				self.jobs += (Job(id=str(int(line[0])+1), processing_times=(int(line[1]), int(line[2])), release_dates=[int(line[3]),int(line[3])+int(line[1])]),)
		if returnList:
			return list(self.jobs)
	
	def sortBy_release(self, jobs: list = None, custom: bool = False, append_stats: bool = False):

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
			seq.sort(key=Job.get_release)
			if append_stats:
				return (seq, min(seq), max(seq))
			return seq

	def sortBy_processingTimes(self, jobs: list = None, custom: bool = False):

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
			seq.sort(key=Job.get_processingTime)
			return seq

	def compare_initial_sols(self, SPT: bool = True, ERD: bool = True):

		spt = list()
		erd = list()

		if SPT:
			spt = self.sortBy_processingTimes()
		if ERD:
			erd = self.sortBy_release()
		
		if SPT and ERD:
			i = 0

			for x in range(len(spt)):
				i += 1 if spt[x].id == erd[x].id else 0
			print(f"{i} matching jobs found")
	
	def set_machines(self, m: int):
		self.machines = m


class Node: 
	"""This class hold the permutation currenlty under analysis and computes the objective function value it yields"""

	def __init__(self, seq: list):
		self.seq = seq
		self.initial_shift: int  = self.seq[0].get_processingTime() + self.seq[0].get_release()
		self.makespan:int  = None
		self.completion: int = None

		for j in self.seq:
			j.reset() # TO-DO: find a clever way to not recompute all this each time we move to a neihborgh


	def eval(self):
		return self.walk_schedule() if self.completion is None else self.completion

	def get_makespan(self):
		return 0 if self.completion is None else self.completion

	def walk_schedule(self):
		"""
		It walks down the sequence of job to compute in linear time schedule parameters

		Objective: C_max
		C[m] = ( n * ( K[m] + p[1][m] )) + SUM(from=1, to=n-1){ (n-i) * ( p[i+1][m] + MAX{ r[i+1][m]-MS(i), 0 ) } }

		K[1] = r[1][1]
		K[m] = r[1][m-1] + p[1][m-1]

		MS[0] = MAX{ 0, r[1][m] }
		MS[1] = MS[0] + p[1][m]
		MS[i] = p[i][m] + MAX { MS[i-1], r[i][m] }
	
		"""
		self.completion = self.seq[0].get_makespan(2)
		self.makespan = self.seq[0].get_makespan(2)

		for i in range(1, len(self.seq)): # first job is already set
			self.seq[i].set_makespan(max(self.seq[i-1].get_makespan(1), self.seq[i].get_release(1)) + self.seq[i].get_processingTime(1), 1)
			self.seq[i].set_release(max(self.seq[i-1].get_makespan(2), self.seq[i].get_makespan(1)), 2)
			self.seq[i].set_makespan(max(self.seq[i-1].get_makespan(2), self.seq[i].get_release(2)) + self.seq[i].get_processingTime(2), 2)
			self.completion += self.seq[i].get_makespan(2)
		self.makespan = self.seq[i].get_makespan(2)

		return self.completion
		
	def is_feasible(self): # NOTE: after the latest release dates all permutations are feasible
		pass


class Explorer:
	"""This object creates head neighborghood after having check feasibility. """

	def __init__(self, instance):
		self.head = instance
		self.neighborghood = set()
	

	def is_feasible():
		pass

def sigmoidFunc(head: Node, neighborgh: Node, temperature: int, minimisation=True) -> float:
	return 1/(1+exp**((head.eval()-neighborgh.eval())/temperature)) if minimisation else 1/(1+exp**(-(head.eval()-neighborgh.eval())/temperature))


def linearCooling(T, cooling_rate: float = 0.95):
	return cooling_rate*T


class SimulatedAnnieling:
	def __init__(self, p: Problem, head: Node=None, T0: int=1000, coolingFunc=None, probaFunc=None):
		self.problem = p
		self.head = self.set_initial_sol(rule="rand") if head is None else head    #  the current node to be evalauted (a feasible permutation of jobs) to be evaluated
		self.T = T0
		self.coolingEngine = linearCooling if coolingFunc is None else coolingFunc
		self.probaEngine = sigmoidFunc if probaFunc is None else probaFunc
		self.explorer = Explorer(self.head)
	

	def set_initial_sol(self, rule: str = "r-max"):
		if rule == "rand":
			seq = list(self.problem.jobs)
			shuffle(seq)

			return Node(seq=seq)

		if rule == "r-max":
			seq = self.problem.sortBy_release()
			ss = seq.pop(len(seq)-1)
			seq.insert(0, ss)
			seq = self.problem.sortBy_processingTimes(jobs=seq[1:])
			seq.insert(0, ss)

			return Node(seq=seq)

		return list() # TO-DO: extend methods to find better initial solutions

	def set_head(self, n: Node):
		self.head = n

	def set_T(self):
		self.T = self.coolingEngine(self.T)

	def move_head(self, to: Node):
		return uniform.rvs() > self.probaEngine(self.head, to=to, temperature=self.T)

p = Problem()
print(p.solver.head.seq ,p.solver.head.eval())