import numpy as np
from process_graph import *

def is_point_on_line(point1, point2, point3):
    # Calculate vectors
    vector1 = point3 - point1
    vector2 = point2 - point1
    
    # Calculate dot products
    dot_product1 = np.dot(vector1, vector2)
    dot_product2 = np.dot(vector2, vector2)
    
    # Calculate parameter
    parameter = dot_product1 / dot_product2
    
    # Check if the parameter is between 0 and 1
    return 0 <= parameter <= 1

def point_within_triangle(A, B, C, P):
    # Define vectors for each triangle edge
    AB = B - A
    BC = C - B
    CA = A - C

    # Define vectors from each vertex to the point
    AP = P - A
    BP = P - B
    CP = P - C

    # Calculate normal vector to the triangle's plane
    normal = np.cross(AB, BC)

    # Check if the point is on the correct side of each edge
    if (np.dot(np.cross(AB, AP), normal) > 0 and
        np.dot(np.cross(BC, BP), normal) > 0 and
        np.dot(np.cross(CA, CP), normal) > 0):
        return True
    else:
        if is_point_on_line(A, B, P) or is_point_on_line(A, C, P) or is_point_on_line(C, B, P):
            return True
        return False



def signed_volume(a, b, c, d):
    return (1/6) * ((b[0]-a[0]) * ((c[1]-a[1])*(d[2]-a[2]) - (d[1]-a[1])*(c[2]-a[2])) +
                    (b[1]-a[1]) * ((c[2]-a[2])*(d[0]-a[0]) - (d[2]-a[2])*(c[0]-a[0])) +
                    (b[2]-a[2]) * ((c[0]-a[0])*(d[1]-a[1]) - (d[0]-a[0])*(c[1]-a[1])))

def check_line_triangle_intersection(A, B, C, P, Q):
    # Calculate signed volumes
    vol1 = signed_volume(A, B, C, P)
    vol2 = signed_volume(A, B, C, Q)

    # Check signs
    if vol1 * vol2 < 0:
        return True  # Intersects
    else:
        return False  # Doesn't intersect

def point_on_triangle_plane(v0, v1, v2, p, tolerance=1e-6):
    # Calculate normal vector of the triangle
    v01 = v1 - v0
    v02 = v2 - v0
    normal = np.cross(v01, v02)
    
    # Check if point p lies in the same plane as the triangle
    v0p = p - v0
    dot_product = np.dot(v0p, normal)
    if np.abs(dot_product) > tolerance:
        return False  # Not in the same plane
    
    # # Check if point p is on an edge
    # on_edge = False
    # for edge in [(v0, v1), (v1, v2), (v2, v0)]:
    #     if is_point_on_line_segment(edge[0], edge[1], p):
    #         on_edge = True
    #         break
    # if on_edge:
    #     return True
    
    # Check if point p is inside the triangle
    # Express p as a linear combination of the edges
    u = np.dot(v0p, v01) / np.dot(v01, v01)
    v = np.dot(v0p, v02) / np.dot(v02, v02)
    if 0 <= u <= 1 and 0 <= v <= 1 and u + v <= 1:
        return True
    else:
        return False

def ray_intersects_triangle(p1, p2, face):
    # break down triangle into the individual points
    v1 = face[0]
    v2 = face[1]
    v3 = face[2]

    ray_vec = p2 - p1

    eps = 0.000001

    # compute edges
    edge1 = v2 - v1
    edge2 = v3 - v1
    pvec = np.cross(ray_vec, edge2)
    det = edge1.dot(pvec)

    if abs(det) < eps:  # no intersection
        return False, None
    inv_det = 1.0 / det
    tvec = p1 - v1
    u = tvec.dot(pvec) * inv_det

    if u < 0.0 or u > 1.0:  # if not intersection
        return False, None

    qvec = np.cross(tvec, edge1)
    v = ray_vec.dot(qvec) * inv_det
    if v < 0.0 or u + v > 1.0:  # if not intersection
        return False, None

    t = edge2.dot(qvec) * inv_det
    if t < eps:
        return False, None

    intersection_point = p1 + t * ray_vec

    if euclidean_distance(p1, intersection_point) < euclidean_distance(p1, p2):
        return True, intersection_point
    
    return False, None

# def ray_intersects_triangle(p1, p2, face):
#     epsilon = 1e-8  # Small positive number for numerical precision

#     # Compute direction vector of the ray
#     direction = p2 - p1

#     # Compute vectors for two edges sharing v0
#     v0 = face[0]
#     v1 = face[1]
#     v2 = face[2]

#     e1 = v1 - v0
#     e2 = v2 - v0

#     # # Compute determinant to check if the ray is parallel to the triangle
#     h = np.cross(direction, e2)
#     a = np.dot(e1, h)
#     if abs(a) < epsilon:
#         return False, None  # Ray is parallel to the triangle

#     # # Compute factors to compute u and v
#     f = 1.0 / a
#     s = p1 - v0
#     u = f * np.dot(s, h)
#     if u < 0 or u > 1:
#         return False, None  # Intersection is outside the triangle

#     q = np.cross(s, e1)
#     v = f * np.dot(direction, q)
#     if v < 0 or u + v > 1:
#         return False, None  # Intersection is outside the triangle

#     # Compute t to find the intersection point
#     t = f * np.dot(e2, q)
#     if t > epsilon:
#         intersection_point = p1 + t * direction
#         min_x, max_x, min_y, max_y, min_z, max_z = 0, 0, 0, 0, 0, 0

#         if p1[0] < p2[0]:
#             min_x = p1[0]
#             max_x = p2[0]
#         else:
#             min_x = p2[0]
#             max_x = p1[0]

#         if p1[1] < p2[1]:
#             min_y = p1[1]
#             max_y = p2[1]
#         else:
#             min_y = p2[1]
#             max_y = p1[1]

#         if p1[2] < p2[2]:
#             min_z = p1[2]
#             max_z = p2[2]
#         else:
#             min_z = p2[2]
#             max_z = p1[2]

#         if min_x <= intersection_point[0] <= max_x and min_y <= intersection_point[1] <= max_y and min_z <= intersection_point[2] <= max_z:
#             return True, intersection_point 

#     return False, None  # No intersection

def find_projection_point_on_line(P1, P2, Q):
    # Compute direction vector of the line
    dir_vec = P2 - P1
    
    # Compute vector from P1 to Q
    P1Q = Q - P1
    
    # Compute projection length
    projection_length = np.dot(P1Q, dir_vec) / np.dot(dir_vec, dir_vec)
    
    # Compute projection point
    projection_point = P1 + projection_length * dir_vec
    
    return projection_point

def select_faces_with_chosen_vertices(vertices, faces, chosen_vertices, loop=2):
    selected_faces = []
    loop_count = 1
    list_vertices = chosen_vertices[:]

    while loop_count <= loop:
        loop_count += 1

        # Convert the chosen vertices list to a set for faster lookup
        chosen_set = set(list_vertices)
        
        # Iterate over each face
        list_vertices = []
        selected_faces = []

        for face in faces:
            # Check if any vertex index in the face is in the chosen set
            if any(vertex_index in chosen_set for vertex_index in face):
                selected_faces.append(face)
                
                for vertex_index in face:
                    list_vertices.append(vertex_index)

                # Convert inner lists to tuples
                tuple_list = [tuple(sublist.tolist()) for sublist in selected_faces]
                unique_tuples = set(tuple_list)
                selected_faces = [np.array(t) for t in unique_tuples]

    selected_faces = vertices[selected_faces]
    return np.array(selected_faces)

