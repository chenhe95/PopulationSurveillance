import math
import random

IMG_WIDTH = 768
IMG_HEIGHT = 576

IMAGE_DIAG = math.sqrt(IMG_HEIGHT^2 + IMG_WIDTH^2)

INTERACTION_VARIANCE = 100^2

BETA = 1
PHI = 1

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

def mu(p1, p2)
	a_i = area_intersection(p1, p2)
	x1, y1, x2, y2 = get_bounding_box(p1)
	a1, b1, a2, b2 = get_bounding_box(p2)

	a_b1 = area_bounding_box(x1, y1, x2, y2)
	a_b2 = area_bounding_box(a1, b1, a2, b2)

	return a_i / (a_b1 + a_b2 - a_i)

def p_pos(p1, p2):
	x, y = get_centroid(p1)
	a, b = get_centroid(p2)

	return math.sqrt((x - a)^2 + (y - b)^2) / IMAGE_DIAG

def p_l(p1, p2):
	return mu(p1, p2) * p_pos(p1, p2)

def psi(p1, p2):

	x, y = get_centroid(p1)
	a, b = get_centroid(p2)

	return 1 - (1 / BETA) * math.exp(-((x - a)^2 + (y - b)^2) / INTERACTION_VARIANCE)

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

def a_ij(proposal, i, j):
	local_score = p_l(proposal[i][0], proposal[j][1])
	global_score = p_g(proposal)

	return global_score / (global_score + local_score)

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

def IMCMC():
	pass