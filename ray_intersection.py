import numpy as np

def ray_intersects_triangle(p1, p2, face):
    epsilon = 1e-6  # Small positive number for numerical precision

    # Compute direction vector of the ray
    direction = p2 - p1

    # Compute vectors for two edges sharing v0
    v0 = face[0]
    v1 = face[1]
    v2 = face[2]
    
    e1 = v1 - v0
    e2 = v2 - v0

    # Compute determinant to check if the ray is parallel to the triangle
    h = np.cross(direction, e2)
    a = np.dot(e1, h)
    if abs(a) < epsilon:
        return False  # Ray is parallel to the triangle

    # Compute factors to compute u and v
    f = 1.0 / a
    s = p1 - v0
    u = f * np.dot(s, h)
    if u < 0 or u > 1:
        return False  # Intersection is outside the triangle

    # Compute q and v
    q = np.cross(s, e1)
    v = f * np.dot(direction, q)
    if v < 0 or u + v > 1:
        return False  # Intersection is outside the triangle

    # Compute t to find the intersection point
    t = f * np.dot(e2, q)
    if t > epsilon:
        intersection_point = p1 + t * direction
        
        # Check if the intersection point lies inside or outside the triangle
        w = 1 - u - v
        if 0 <= u <= 1 and 0 <= v <= 1 and 0 <= w <= 1:
            if 0 <= t <= np.linalg.norm(p2 - p1):
                return True

    return False  # No intersection

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

def select_faces_with_chosen_vertices(vertices, faces, chosen_vertices):
    selected_faces = []
    
    # Convert the chosen vertices list to a set for faster lookup
    chosen_set = set(chosen_vertices)
    
    # Iterate over each face
    for face in faces:
        # Check if any vertex index in the face is in the chosen set
        if any(vertex_index in chosen_set for vertex_index in face):
            selected_face = vertices[face]
            selected_faces.append(selected_face)
    
    return np.array(selected_faces)

vertices = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
    [13, 14, 15],
    [16, 17, 18],
])

faces = np.array([
    [0, 1, 2],
    [1, 2, 3],
    [3, 4, 5]
])

chosen_vertices = [1, 4]

print(select_faces_with_chosen_vertices(vertices, faces, chosen_vertices))
# p1 = np.array([1.5, 1.5, -1])
# p2s = np.array([
#     [1.5, 1.5, 1],
#     [0, 1.5, 0],
#     [1.5, 1.5, -0.5]
# ])
# v0 = np.array([1, 1, 0])
# v1 = np.array([2, 1, 0])
# v2 = np.array([1, 2, 0])

# for p2 in p2s:
#     print(ray_intersects_triangle(p1, p2, v0, v1, v2))
    
    
# projection_point = find_projection_point_on_line(v0, v2, np.array([1, 1.5, -1]))
# print(projection_point)
