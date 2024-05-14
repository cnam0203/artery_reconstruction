from shapely.geometry import LineString, Polygon
import numpy as np
l = LineString([np.array([1,0.5,0.5]),np.array([3,0.5,0.5])])
p = Polygon([np.array([1.2,0.0,0.]),np.array([2.2,1.0,0.]),np.array([2.8,0.5,1.])])
print(l.intersection(p))