from copy import copy

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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


def rotation_matrix(alpha, beta, gamma):
    """Erstellt eine Rotationsmatrix aus Euler-Winkeln."""
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])

    R_y = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])

    R_z = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])

    return R_z @ R_y @ R_x


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

    ax.add_collection3d(Poly3DCollection(faces, alpha=0.5,  facecolors='g',edgecolor='k'))
    ax.add_collection3d(Poly3DCollection(plane, alpha=0.5, edgecolor='k'))


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_title(title)

    plt.show()

def plot_cube_2d(vertices, title="2D Rotierter Würfel", plane=None):
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

    plt.show()

def sort_vertices_by_z(vertices):
    """Gibt die Ecken des rotierten Würfels sortiert nach der Z-Koordinate zurück."""
    indices = np.argsort(vertices[:, 2])  # Indizes nach der Z-Koordinate sortieren
    sorted_vertices = vertices[indices]
    return sorted_vertices, indices


def plot_cube_zoom_corner(vertices, title="Zoom Corner", plane=None):
    """Plottet den Würfel im 3D-Raum und optional eine Ebene."""
    sorted_vertices, indices = sort_vertices_by_z(vertices)

    lowest_corner = indices[0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    vertices2d = copy(vertices)
    for i in range(len(vertices2d)):
        vertices2d[i][2] = -1

    faces2d = [[vertices2d[j] for j in face] for face in
             [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
              (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)] if lowest_corner in face]

    ax.add_collection3d(Poly3DCollection(faces2d, alpha=0.5, edgecolor='k'))

    print(f"smallest corner: {sorted_vertices[0]}")
    x_lim = sorted_vertices[0][0] + 0.1, sorted_vertices[0][0] - 0.1
    y_lim = sorted_vertices[0][1] + 0.1, sorted_vertices[0][1] - 0.1

    elev = 90
    azim = 0

    ax.view_init(elev=elev, azim=azim)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim([-1, 1])
    ax.set_title(title)

    plt.show()

def compute_plane_from_two_lowest(vertices):
    """Berechnet die Ebene zwischen den zwei niedrigsten Ecken."""
    h1, h2  = np.sort(vertices[:, 2])[:2]

    z_height = ((h1 - h2) / 2) + h2

    # Define the four corners of the plane
    x_min, x_max = -1, 1
    y_min, y_max = -1, 1

    plane_vertices = [
        [x_min, y_min, z_height],
        [x_max, y_min, z_height],
        [x_max, y_max, z_height],
        [x_min, y_max, z_height]
    ]
    plane_faces = [plane_vertices]
    return plane_faces

# Setze eine zufällige Rotation
true_alpha, true_beta, true_gamma = np.radians([30, 45, 20])
rotated_vertices = rotated_cube_vertices(true_alpha, true_beta, true_gamma)

# Berechne die Ebene zwischen den zwei niedrigsten Ecken
equation_plane = compute_plane_from_two_lowest(rotated_vertices)

# Plotte den Würfel mit der berechneten Ebene
plot_cube(rotated_vertices, title="Original Rotierter Würfel mit Ebene", plane=equation_plane)
plot_cube_2d(rotated_vertices, title="2D Projektion", plane=equation_plane)
plot_cube_zoom_corner(rotated_vertices, title="2D Projektion Zoom", plane=equation_plane)

# Projiziere den Würfel auf die XY-Ebene
# project_cube_to_xy(rotated_vertices)
