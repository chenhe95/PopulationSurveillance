import math

IMG_WIDTH = 768
IMG_HEIGHT = 576

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

	return max(x1, a1), max(y1, b1), min(x2, a2), min(y2, b2)

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

	return math.sqrt((x - a)^2 + (y - b)^2) / math.sqrt(IMG_HEIGHT^2 + IMG_WIDTH^2)

def p_l(p1, p2):
	return mu(p1, p2) * p_pos(p1, p2)



