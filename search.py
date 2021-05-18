def basic_search(target, vector: list, use_sort=False, use_py_sort=False):
	i=0
	while i < len(vector):
		if vector[i] == target:
			return True
		i += 1
	return False


def bin_search(target, vector: list, use_sort=False, use_py_sort=False):
	start = 0
	end = len(vector)-1
	i = (start+end)/2

	if vector[i] == target:
		return True
	if vector[i] > target:
		bin_search(target, vector[:end])
	bin_search(target, vector[start:])
