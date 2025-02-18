from copy import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from networkx.classes import neighbors
from scipy.spatial.transform import Rotation as R

matplotlib.use('TkAgg')
plt.ion()

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

block = True

# Definiere die ursprünglichen Eckpunkte eines Würfels mit Kantenlänge 1
cube_vertices = np.array([
    [-0.5, -0.5, -0.5],
    [0.5, -0.5, -0.5],
    [-0.5, 0.5, -0.5],
    [0.5, 0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [0.5, -0.5, 0.5],
    [-0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5]
])

cube_edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]



def rotation_matrix(alpha, beta, gamma, degrees=True):
    """Creates a rotation matrix from Euler angles using SciPy."""
    return R.from_euler('xyz', [alpha, beta, gamma], degrees=degrees).as_matrix()


def rotated_cube_vertices(alpha, beta, gamma):
    """Gibt die transformierten Koordinaten des Würfels nach Rotation zurück."""
    R = rotation_matrix(alpha, beta, gamma)
    return np.dot(cube_vertices, R.T)


def plot_cube(vertices, title="3D Rotierter Würfel", plane=None):
    """Plottet den Würfel im 3D-Raum und optional eine Ebene."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    faces = [[vertices[j] for j in face] for face in
             [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
              (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]]

    ax.add_collection3d(Poly3DCollection(faces, alpha=0.5, facecolors='g', edgecolor='k'))
    ax.add_collection3d(Poly3DCollection(plane, alpha=0.5, edgecolor='k'))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_title(title)

    plt.show(block=block)


def plot_cube_2d(vertices, title="2D Rotierter Würfel"):
    """Plottet den Würfel im 3D-Raum und optional eine Ebene."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    vertices2d = copy(vertices)
    for i in range(len(vertices2d)):
        vertices2d[i][2] = -1

    faces2d = [[vertices2d[j] for j in face] for face in
               [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
                (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]]

    faces = [[vertices[j] for j in face] for face in
             [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
              (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]]

    ax.add_collection3d(Poly3DCollection(faces2d, alpha=0.5, edgecolor='k'))
    ax.add_collection3d(Poly3DCollection(faces, alpha=0.5, edgecolor='k'))
    elev = 90
    azim = 0

    ax.view_init(elev=elev, azim=azim)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_title(title)

    plt.show(block=block)


def sort_vertices_by_z(vertices):
    """Gibt die Ecken des rotierten Würfels sortiert nach der Z-Koordinate zurück."""
    indices = np.argsort(vertices[:, 2])  # Indizes nach der Z-Koordinate sortieren
    sorted_vertices = vertices[indices]
    return sorted_vertices, indices


def plot_cube_zoom_corner(vertices, title="Zoom Corner"):
    sorted_vertices, indices = sort_vertices_by_z(vertices)
    lowest_corner = indices[0]

    fig, ax = plt.subplots()
    edges = np.array([np.array(e) for e in cube_edges if lowest_corner in e]).flatten()
    edges = np.delete(edges, np.argwhere(lowest_corner))
    vertices2d = vertices[:, :2]

    lines = [(vertices2d[lowest_corner], vertices2d[i]) for i in edges]
    for p1, p2 in lines:
        x_values = [p1[0], p2[0]]
        y_values = [p1[1], p2[1]]
        ax.plot(x_values, y_values, color='black')


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    plt.grid()

    plt.show()


def plot_cube_zoom_with_plane_intersection(vertices, title="Zoom Corner", z_plane=0):
    sorted_vertices, indices = sort_vertices_by_z(vertices)
    lowest_corner = indices[0]

    fig, ax = plt.subplots()
    edges = np.array([np.array(e) for e in cube_edges if lowest_corner in e]).flatten()
    edges = np.delete(edges, np.argwhere(edges == lowest_corner))
    vertices2d = vertices[:, :2]

    lines = [(vertices2d[lowest_corner], vertices2d[i]) for i in edges]
    lines_3d = [(vertices[lowest_corner], vertices[i]) for i in edges]
    for p1, p2 in lines:
        x_values = [p1[0], p2[0]]
        y_values = [p1[1], p2[1]]
        ax.plot(x_values, y_values, color='black')

    for p1, p2 in lines_3d:
        x, y, _ = line_plane_intersection(p1, p2, z_plane)
        ax.plot(x, y, 'rx')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    plt.grid()

    plt.show()


def plot_imprint(vertices, title="Imprint", z_plane=0):
    fig, ax = plt.subplots()
    lines = get_imprint_lines(vertices, z_plane)
    for p1, p2 in lines:
        x_values = [p1[0], p2[0]]
        y_values = [p1[1], p2[1]]
        ax.plot(x_values, y_values, color='black')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    plt.grid()

    plt.show()


def get_imprint_lines(vertices, z_plane):
    sorted_vertices, indices = sort_vertices_by_z(vertices)
    lowest_corner = indices[0]

    edges = np.array([np.array(e) for e in cube_edges if lowest_corner in e]).flatten()
    edges = np.delete(edges, np.argwhere(edges == lowest_corner))
    vertices2d = vertices[:, :2]
    lines_3d = [(vertices[lowest_corner], vertices[i]) for i in edges]
    lines = []
    for p1, p2 in lines_3d:
        x, y, _ = line_plane_intersection(p1, p2, z_plane)
        lines.append((vertices2d[lowest_corner], (x, y)))
    return lines


def get_plane_vertices(z_plane: float):
    """Berechnet die Ebene zwischen den zwei niedrigsten Ecken."""
    # Define the four corners of the plane
    x_min, x_max = -1, 1
    y_min, y_max = -1, 1

    plane_vertices = [
        [x_min, y_min, z_plane],
        [x_max, y_min, z_plane],
        [x_max, y_max, z_plane],
        [x_min, y_max, z_plane]
    ]
    plane_faces = [plane_vertices]
    return plane_faces


def get_plane_between_corners(vertices: np.ndarray):
    h1, h2 = np.sort(vertices[:, 2])[:2]
    z_height = ((h1 - h2) / 2) + h2
    return z_height


def line_plane_intersection(p1, p2, z_plane):
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    if z1 == z2:  # The line is parallel to the plane
        return None if z1 != z_plane else (x1, y1, z_plane)

    t = (z_plane - z1) / (z2 - z1)

    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    return x, y, z_plane


def calculate_angle(line1, line2):
    p1, p2 = line1
    p3, p4 = line2

    v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    v2 = np.array([p4[0] - p3[0], p4[1] - p3[1]])

    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to handle numerical errors

    return np.degrees(angle)


def sort_lines_counterclockwise(lines):
    center = np.mean([p for line in lines for p in line], axis=0)

    def angle_from_center(line):
        mid_point = np.mean(line, axis=0)
        return np.arctan2(mid_point[1] - center[1], mid_point[0] - center[0])

    return sorted(lines, key=angle_from_center)


def get_imprint_angle(vertices):
    z_height = get_plane_between_corners(rotated_vertices)
    lines = get_imprint_lines(vertices, z_height)
    lines = sort_lines_counterclockwise(lines)
    alpha = calculate_angle(lines[0], lines[1])
    beta = calculate_angle(lines[1], lines[2])
    gamma = calculate_angle(lines[0], lines[2])

    x_axis =  np.array([[0, 0], [1,0]])
    # alpha = calculate_angle(lines[0], x_axis)
    # beta = calculate_angle(lines[1], x_axis)
    # gamma = calculate_angle(lines[2], x_axis)
    return alpha, beta, gamma


def generate_cube_symmetry_rotations():
    """
    Generate the 24 unique rotation matrices that map a cube onto itself.
    """
    cube_symmetries = []
    base_rotations = [
        (0, 0, 0), (90, 0, 0), (180, 0, 0), (270, 0, 0),  # Rotations around X-axis
        (0, 90, 0), (0, 270, 0),  # Rotations around Y-axis
        (0, 0, 90), (0, 0, 270),  # Rotations around Z-axis
        (90, 90, 0), (90, 270, 0), (270, 90, 0), (270, 270, 0),  # Diagonal rotations
        (90, 0, 90), (90, 0, 270), (270, 0, 90), (270, 0, 270),  # Edge-based rotations
        (180, 90, 0), (180, 270, 0), (180, 0, 90), (180, 0, 270)  # Additional flips
    ]
    for angles in base_rotations:
        cube_symmetries.append(R.from_euler('xyz', angles, degrees=True).as_matrix())

    return cube_symmetries


def are_cube_rotations_equivalent(euler1, euler2, degrees=True, tol=1e-2):
    """
    Check if two rotations (given in Euler angles) are equivalent under cube symmetry.

    Parameters:
    - euler1: tuple/list of (yaw, pitch, roll) angles for rotation 1
    - euler2: tuple/list of (yaw, pitch, roll) angles for rotation 2
    - degrees: whether angles are in degrees (default True)
    - tol: numerical tolerance for matrix comparison

    Returns:
    - True if the two rotations are equivalent under cube symmetry, False otherwise.
    """
    # Convert Euler angles to rotation matrices
    R1 = R.from_euler('xyz', euler1, degrees=degrees).as_matrix()
    R2 = R.from_euler('xyz', euler2, degrees=degrees).as_matrix()

    # Get all cube symmetry transformations
    cube_symmetries = generate_cube_symmetry_rotations()

    # Check if R1 matches any rotation of R2 under cube symmetry
    for S in cube_symmetries:
        if np.allclose(R1, S @ R2, atol=tol):
            return True  # Found an equivalent rotation
    return False

def objective_function(rotation_angles, imprint_angles, vectors):
    rx, ry, rz = rotation_angles
    rotated_vertices = rotated_cube_vertices(rx, ry, rz)

    calc_vectors = get_corner_vectors(rotated_vertices)
    #print(f"Rotation Angles: {rx}, {ry}, {rz}")
    z_height = get_plane_between_corners(rotated_vertices)
    lines = get_imprint_lines(rotated_vertices, z_height)
    lines = sort_lines_counterclockwise(lines)
    imprint_alpha = calculate_angle(lines[0], lines[1])
    imprint_beta = calculate_angle(lines[1], lines[2])
    imprint_gamma = calculate_angle(lines[0], lines[2])
    #print(f"Calculated Angles: alpha={imprint_alpha}, beta={imprint_beta}, gamma={imprint_gamma}")

    a,b,g = imprint_angles

    error_a = np.linalg.norm(imprint_alpha - a)
    error_b = np.linalg.norm(imprint_beta - b)
    error_g = np.linalg.norm(imprint_gamma - g)

    error_vec = 0

    for i in range(len(vectors)):
        error_vec += (calc_vectors[i] - vectors[i])**2

    #print(f"Errors: {error_a}, {error_b}, {error_g}")

    return error_a + error_b + error_g + error_vec

def find_cube_rotation(imprint_angles, vectors):
    from scipy.optimize import minimize

    """Findet die optimalen Rotationswinkel für den Würfel."""
    initial_guess = np.array([1, 1, 1])  # Startwerte für die Winkel
    result = minimize(objective_function, initial_guess, args=(imprint_angles, vectors,), method='Nelder-Mead')
    return result.x


def get_point_of_interest(vertices):
    sorted_vertices, indices = sort_vertices_by_z(vertices)
    lowest_corner = indices[0]

    edges = np.array([np.array(e) for e in cube_edges if lowest_corner in e]).flatten()
    edges = np.delete(edges, np.argwhere(edges == lowest_corner))
    vertices2d = vertices[:, :2]
    center = vertices2d[lowest_corner]
    neighbor = [vertices2d[i] for i in edges]
    return center, neighbor

def get_corner_vectors(vertices):
    center, neighbor = get_point_of_interest(vertices)
    vectors = 0
    for n in neighbor:
        v = n - center
        norm = np.linalg.norm(v)
        vectors += v / norm if norm != 0 else v
    return vectors


# Setze eine zufällige Rotation
true_alpha, true_beta, true_gamma = np.array([-45, 35.264, 0])
true_alpha_2, true_beta_2, true_gamma_2 = np.array([45.  ,  35.26 ,-52.48])
rotated_vertices = rotated_cube_vertices(true_alpha, true_beta, true_gamma)
rotated_vertices_2 = rotated_cube_vertices(true_alpha_2, true_beta_2, true_gamma_2)
equivialent = are_cube_rotations_equivalent((true_alpha, true_beta, true_gamma), (true_alpha_2, true_beta_2, true_gamma_2))
print(f"are rotations equivalent: {equivialent}")

# Berechne die Ebene zwischen den zwei niedrigsten Ecken
z_height = get_plane_between_corners(rotated_vertices)
equation_plane = get_plane_vertices(z_height)
imprint_angles = get_imprint_angle(rotated_vertices)
rotated_vectors = get_corner_vectors(rotated_vertices)
print(f"imprint angles: {imprint_angles}")
calc_angles = np.round(find_cube_rotation(imprint_angles, rotated_vectors), 2)
np.set_printoptions(suppress=True)
print(f"Calculated Angles: {calc_angles}")
calc_alpha, calc_beta, calc_gamma = np.array(calc_angles)
calc_vertices = rotated_cube_vertices(calc_alpha, calc_beta, calc_gamma)
z_height_2 = get_plane_between_corners(calc_vertices)
equivialent = are_cube_rotations_equivalent((true_alpha, true_beta, true_gamma), (calc_alpha, calc_beta, calc_gamma))
print(f"are rotations equivalent: {equivialent}")

plot = False
plot_imprint(rotated_vertices, title="Imprint", z_plane=z_height)
plot_imprint(calc_vertices, title="Imprint", z_plane=z_height_2)
if plot:
    # Plotte den Würfel mit der berechneten Ebene
    plot_cube(rotated_vertices, title="Original Rotierter Würfel mit Ebene", plane=equation_plane)
    plot_cube_zoom_corner(rotated_vertices, title="2D Projektion Zoom")
    plot_cube_zoom_with_plane_intersection(rotated_vertices, title="2D Projektion Zoom", z_plane=z_height)
    plot_cube_2d(rotated_vertices, title="2D Projektion")
