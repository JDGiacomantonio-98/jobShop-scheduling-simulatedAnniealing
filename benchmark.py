from matplotlib import pyplot as plt
from datetime import datetime as dt
from random import randint
from os import getcwd, path, mkdir

from search import basic_search, bin_search

def plot_time(search_func,  target: int=randint(0, dt.now().microsecond), grow_n_to: int=100, worse_case: bool=False, save: bool=True):
	values = {
		'x': list(),
		'y': list()
	}
	n=1
	r=0

	while n <= grow_n_to:
		t=dt.now()
		r +=1 if search_func(target=target if not worse_case else target+grow_n_to+1, vector=[randint(0, target+grow_n_to) for x in range(n)]) else 0
		t=dt.now()-t

		values['y'].append(t.total_seconds())
		values['x'].append(n)

		n += 1
	
	plt.plot(values['x'], values['y'])
	plt.show()
	print(f'# of hits: {r}')
	if save:
		if not path.isdir(f'{getcwd()}\\performance-graphs'):
			mkdir(f'{getcwd()}\\performance-graphs')
		plt.savefig(f'{getcwd()}\\performance-graphs\\{search_func.__name__}-{grow_n_to}  {dt.now().isocalendar()}')


def map(sizes: tuple=(10, 100, 1000, 5000, 10000)):
	for n in sizes:
		plot_time(basic_search, grow_n_to=n, worse_case=True)
