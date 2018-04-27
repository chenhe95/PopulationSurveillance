import math
import random
import numpy as np
import pickle

from obj_tracking import generate_video

IMG_WIDTH = 768
IMG_HEIGHT = 576

IMAGE_DIAG = math.sqrt(IMG_HEIGHT ** 2 + IMG_WIDTH ** 2)

INTERACTION_VARIANCE = 100 ** 2 # 100 ** 2 in paper
DISTANCE_FACTOR = 2 # 25
GAMMA = 1 # 0.2 in paper
BETA = 1 # 1 in paper
PHI = 1

MH_ITER_N = 1000

def person_equals(obj1, obj2):
	return obj1["topleft"] == obj2["topleft"] and obj1["bottomright"] == obj2["bottomright"] and obj1["label"] == "person" and obj2["label"] == "person"

def get_bounding_box(obj):
	# FORMAT
	# {'topleft': {'y': 163, 'x': 489}, 'confidence': 0.74837613, 'bottomright': {'y': 240, 'x': 517}, 'label': 'person'}
	tl = obj["topleft"]
	br = obj["bottomright"]

	return tl["x"], tl["y"], br["x"], br["y"]

def area_bounding_box(x1, y1, x2, y2):
	return (x2 - x1) * (y2 - y1)

def get_centroid(obj):
	x1, y1, x2, y2 = get_bounding_box(obj)

	return (x1 + x2) / 2, (y1 + y2) / 2

def bounding_box_intersection(obj1, obj2):
	x1, y1, x2, y2 = get_bounding_box(obj1)
	a1, b1, a2, b2 = get_bounding_box(obj2)

	x, y, a, b = max(x1, a1), max(y1, b1), min(x2, a2), min(y2, b2)

	if a <= x or b <= y:
		return 0, 0, 0, 0
	else:
		return x, y, a, b

def area_intersection(obj1, obj2):
	x1, y1, x2, y2 = bounding_box_intersection(obj1, obj2)
	return area_bounding_box(x1, y1, x2, y2)

def soft_area_intersection(obj1, obj2):
	x1, y1, x2, y2 = bounding_box_intersection(obj1, obj2)
	return max(1, area_bounding_box(x1, y1, x2, y2))

def mu(p1, p2):
	a_i = soft_area_intersection(p1, p2)
	x1, y1, x2, y2 = get_bounding_box(p1)
	a1, b1, a2, b2 = get_bounding_box(p2)

	a_b1 = area_bounding_box(x1, y1, x2, y2)
	a_b2 = area_bounding_box(a1, b1, a2, b2)

	return a_i / (a_b1 + a_b2 - a_i)

def p_pos(p1, p2):
	x, y = get_centroid(p1)
	a, b = get_centroid(p2)
	# return 1 / (math.sqrt((x - a) ** 2 + (y - b) ** 2) + DISTANCE_FACTOR)
	return psi(p1, p2)

def p_l(p1, p2):
	if p1 is None or p2 is None:
		return 0
	return mu(p1, p2) * p_pos(p1, p2)

def psi(p1, p2):
	if p1 is None or p2 is None:
		return 1 # before 1 - 1 / BETA
	x, y = get_centroid(p1)
	a, b = get_centroid(p2)

	return 1 - (1 / BETA) * math.exp(-((x - a) ** 2 + (y - b) ** 2) / INTERACTION_VARIANCE)

def psi_multiplicative_sum(proposal):
	# PROPOSAL FORMAT
	# [[p11, p21], ..., [p1n, p2n]]
	psi_ms = 1
	for p1p2 in proposal:
		psi_ms = psi_ms * psi(p1p2[0], p1p2[1])

	return psi_ms

def p_appearance_phi():
	# p_appearance * phi
	return 1 * PHI

def swap_proposal(proposal, i, j):
	temp = proposal[j][1]
	proposal[j][1] = proposal[i][1]
	proposal[i][1] = temp

	return proposal

def p_g(proposal):
	return psi_multiplicative_sum(proposal) * p_appearance_phi()

def copy_proposal(proposal):
	new_proposal = [[proposal[i][0], proposal[i][1]] for i in xrange(len(proposal))]
	return new_proposal

def dist_metric_l2(p1, p2):
	return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def dist_metric_gaussian(p1, p2, variance=INTERACTION_VARIANCE):
	return 1 / math.sqrt(2 * math.pi) * math.exp(dist_metric_l2(p1, p2) / variance)

def dist_metric_exp(p1, p2):
	return math.exp(dist_metric_l2(p1, p2) / INTERACTION_VARIANCE)

def a_ij_local(proposal, i, j):
	local_score_old = p_l(proposal[i][0], proposal[i][1])
	local_score_new = p_l(proposal[i][0], proposal[j][1])

	if local_score_old == 0: 
		if local_score_new > 0.0000001:
			return local_score_new
		return 0

	return local_score_new / local_score_old

def a_ij_global(proposal, i, j):
	local_score_old = p_g(proposal)
	c_proposal = copy_proposal(proposal)
	c_proposal = swap_proposal(c_proposal, i, j)
	local_score_new = p_g(c_proposal)

	if local_score_old == 0: 
		if local_score_new > 0.0000001:
			return local_score_new
		return 0

	return local_score_new / local_score_old

def a_ij(proposal, i, j):
	local_score = p_l(proposal[i][0], proposal[j][1])
	global_score = p_g(proposal)

	if abs(global_score + local_score) < 10e-10:
		return 0

	print "global score, local score = ", str(global_score), ", ", str(local_score)

	return global_score / (global_score + local_score)


def a_ij_dist(proposal, i, j, dist_metric):
	current_score = 0
	proposal_score = 0

	c_proposal = copy_proposal(proposal)
	c_proposal = swap_proposal(c_proposal, i, j)

	for p in proposal:

		if p[0] is None or p[1] is None:
			continue

		x1, y1 = get_centroid(p[0])
		x2, y2 = get_centroid(p[1])

		current_score = current_score + dist_metric((x1, y1), (x2, y2))

	for p in c_proposal:

		if p[0] is None or p[1] is None:
			continue

		x1, y1 = get_centroid(p[0])
		x2, y2 = get_centroid(p[1])

		proposal_score = proposal_score + dist_metric((x1, y1), (x2, y2))

	if proposal_score < 0.0000000001:
		return 1

	return current_score / proposal_score

def a_ij_y(proposal, i, j):
	current_score = 0
	proposal_score = 0

	c_proposal = copy_proposal(proposal)
	c_proposal = swap_proposal(c_proposal, i, j)

	if proposal[i][0] is None or proposal[j][1] is None:
		return 0.5 # ???

	x1, y1 = get_centroid(proposal[i][0])
	x2, y2 = get_centroid(proposal[j][1])

	current_score = dist_metric_l2((x1, y1), (x2, y2))

	for p in c_proposal:

		if p[0] is None or p[1] is None:
			continue

		x1, y1 = get_centroid(p[0])
		x2, y2 = get_centroid(p[1])

		proposal_score = proposal_score + dist_metric_l2((x1, y1), (x2, y2))

	if proposal_score < 0.0000000001:
		return 1

	return 1 - current_score / proposal_score

def a_ij_hardcap(proposal, i, j, hardcap = 30):
	current_score = 0
	proposal_score = 0

	c_proposal = copy_proposal(proposal)
	c_proposal = swap_proposal(c_proposal, i, j)

	if proposal[i][0] is None or proposal[j][1] is None:
		return 0.5 # ???

	x1, y1 = get_centroid(proposal[i][0])
	x2, y2 = get_centroid(proposal[j][1])

	current_score = dist_metric_l2((x1, y1), (x2, y2))

	if current_score >= hardcap:
		return 0

	return a_ij_dist(proposal, i, j, dist_metric_exp)

def filter_people(di):
	return filter(lambda x: x["label"] == "person", di)

def generate_random_proposal(d, i, j):
	# d is loaded from the time series pickle file
	# Generates proposals for d[i] X d[j]
	if i + 1 >= len(d):
		return None 
	di = filter_people(d[i])
	dj = filter_people(d[j])

	maxlen = max(len(di), len(dj))
	proposal = [None for i in xrange(maxlen)]

	for k in xrange(maxlen):
		if k >= len(di):
			proposal[k] = [None, dj[k]]
		elif k >= len(dj):
			proposal[k] = [di[k], None]
		else:
			proposal[k] = [di[k], dj[k]]

	return proposal

def IMCMC(d, t):
	proposal = generate_random_proposal(d, t, t + 1)
	iteration_N = MH_ITER_N
	lp = len(proposal)
	for _ in xrange(iteration_N):

		for i in xrange(lp):
			p_l_distribution = np.array([p_l(proposal[i][0], proposal[j][1]) for j in xrange(lp)])
			if sum(p_l_distribution) == 0:
				return proposal
			p_l_distribution = p_l_distribution / sum(p_l_distribution)
			p_l_values = range(lp)

			j = np.random.choice(p_l_values, 1, p=p_l_distribution)[0]

			rand = random.random()
			if rand < GAMMA:
				rand_2 = random.random()
				alpha = a_ij(proposal, i, j)

				if rand_2 < alpha:
					proposal = swap_proposal(proposal, i, j)
				else:
					# Sample a new swap using distribution (8)
					# and accept with probability (9)
					pass

	return proposal			

def IMCMC_local(d, t):
	proposal = generate_random_proposal(d, t, t + 1)
	iteration_N = MH_ITER_N
	lp = len(proposal)
	for _ in xrange(iteration_N):
		p_l_distribution = -np.array([p_l(proposal[i][0], proposal[j][1]) for j in xrange(lp) for i in xrange(lp)]) + 1.0
		if sum(p_l_distribution) == 0:
			return proposal

		p_l_distribution = p_l_distribution / sum(p_l_distribution)

		p_l_values = [[i, j] for j in xrange(lp) for i in xrange(lp)]

		swap_selection = np.random.choice(range(len(p_l_values)), 1, p=p_l_distribution)[0]
		i, j = p_l_values[swap_selection][0], p_l_values[swap_selection][1]
		rand = random.random()
		alpha = a_ij_local(proposal, i, j)
		if rand < alpha:
			proposal = swap_proposal(proposal, i, j)

	return proposal		

def IMCMC_global(d, t):
	proposal = generate_random_proposal(d, t, t + 1)
	iteration_N = MH_ITER_N
	lp = len(proposal)
	for _ in xrange(iteration_N):
		p_l_distribution = -np.array([p_l(proposal[i][0], proposal[j][1]) for j in xrange(lp) for i in xrange(lp)]) + 1.0
		if sum(p_l_distribution) == 0:
			return proposal

		p_l_distribution = p_l_distribution / sum(p_l_distribution)

		p_l_values = [[i, j] for j in xrange(lp) for i in xrange(lp)]

		swap_selection = np.random.choice(range(len(p_l_values)), 1, p=p_l_distribution)[0]
		i, j = p_l_values[swap_selection][0], p_l_values[swap_selection][1]
		rand = random.random()
		alpha = a_ij_global(proposal, i, j)
		if rand < alpha:
			proposal = swap_proposal(proposal, i, j)

	return proposal		


def IMCMC_dist(d, t):
	proposal = generate_random_proposal(d, t, t + 1)
	iteration_N = MH_ITER_N
	lp = len(proposal)
	for _ in xrange(iteration_N):
		# p_l_distribution = -np.array([p_l(proposal[i][0], proposal[j][1]) for j in xrange(lp) for i in xrange(lp)]) + 1.0

		p_l_distribution = np.array([1.0 / len(proposal) for j in xrange(lp) for i in xrange(lp)]) 

		if sum(p_l_distribution) == 0:
			return proposal

		p_l_distribution = p_l_distribution / sum(p_l_distribution)

		p_l_values = [[i, j] for j in xrange(lp) for i in xrange(lp)]

		swap_selection = np.random.choice(range(len(p_l_values)), 1, p=p_l_distribution)[0]
		i, j = p_l_values[swap_selection][0], p_l_values[swap_selection][1]
		rand = random.random()
		alpha = a_ij_dist(proposal, i, j, dist_metric_exp)

		# print alpha

		if rand < alpha:
			proposal = swap_proposal(proposal, i, j)

	return proposal		

def IMCMC_dist_y(d, t):
	proposal = generate_random_proposal(d, t, t + 1)
	iteration_N = MH_ITER_N
	lp = len(proposal)
	for _ in xrange(iteration_N):
		# p_l_distribution = -np.array([p_l(proposal[i][0], proposal[j][1]) for j in xrange(lp) for i in xrange(lp)]) + 1.0

		p_l_distribution = np.array([1.0 / len(proposal) for j in xrange(lp) for i in xrange(lp)]) 

		if sum(p_l_distribution) == 0:
			return proposal

		p_l_distribution = p_l_distribution / sum(p_l_distribution)

		p_l_values = [[i, j] for j in xrange(lp) for i in xrange(lp)]

		swap_selection = np.random.choice(range(len(p_l_values)), 1, p=p_l_distribution)[0]
		i, j = p_l_values[swap_selection][0], p_l_values[swap_selection][1]
		rand = random.random()
		alpha = a_ij_y(proposal, i, j)

		# print alpha

		if rand < alpha:
			proposal = swap_proposal(proposal, i, j)

	return proposal		

def IMCMC_hardcap(d, t):
	proposal = generate_random_proposal(d, t, t + 1)
	iteration_N = MH_ITER_N
	lp = len(proposal)
	for _ in xrange(iteration_N):
		# p_l_distribution = -np.array([p_l(proposal[i][0], proposal[j][1]) for j in xrange(lp) for i in xrange(lp)]) + 1.0

		p_l_distribution = np.array([1.0 / len(proposal) for j in xrange(lp) for i in xrange(lp)]) 

		if sum(p_l_distribution) == 0:
			return proposal

		p_l_distribution = p_l_distribution / sum(p_l_distribution)

		p_l_values = [[i, j] for j in xrange(lp) for i in xrange(lp)]

		swap_selection = np.random.choice(range(len(p_l_values)), 1, p=p_l_distribution)[0]
		i, j = p_l_values[swap_selection][0], p_l_values[swap_selection][1]
		rand = random.random()
		alpha = a_ij_hardcap(proposal, i, j)

		# print alpha

		if rand < alpha:
			proposal = swap_proposal(proposal, i, j)

	return proposal		


def metropolis_hastings(d):
	max_time = len(d) - 1
	proposals = [None for i in xrange(max_time)]

	for t in xrange(max_time):
		proposals[t] = IMCMC_hardcap(d, t)

	return proposals

if __name__ == "__main__":
	d = None
	with open("filtered_obj_1_25.pkl", "r") as f_in:
		d = pickle.load(f_in)

	proposals = metropolis_hastings(d)

	print "Finished computing metropolis hastings"

	print proposals[0]
	print proposals[1]

	generate_video(proposals, 1, "vid_test.avi")