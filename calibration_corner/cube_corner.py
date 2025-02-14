import numpy as np
from scipy.optimize import minimize
import math

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


def objective_function(angles, known_x, known_y, cut_points):
    """Optimierungsfunktion zur Minimierung der Abweichung der unteren Ecke und der Schnittpunkte."""
    alpha, beta, gamma = angles
    rotated_vertices = rotated_cube_vertices(alpha, beta, gamma)

    # Finde die unterste Ecke (minimale z-Koordinate)
    min_z_idx = np.argmin(rotated_vertices[:, 2])
    min_vertex = rotated_vertices[min_z_idx]

    # Berechne die Kanten, die von der horizontalen Ebene geschnitten werden
    edges = [(0, 1), (0, 2), (0, 4)]  # Beispiel für relevante Kanten
    intersection_errors = 0

    for i, (v1, v2) in enumerate(edges):
        p1, p2 = rotated_vertices[v1], rotated_vertices[v2]
        t = -p1[2] / (p2[2] - p1[2])  # Parameter für lineare Interpolation
        intersection = p1 + t * (p2 - p1)  # Schnittpunkt der Kante mit der z=0 Ebene
        intersection_errors += np.sum((intersection[:2] - cut_points[i][:2]) ** 2)  # Fehler in x und y

    # Fehler zwischen bekannten und berechneten Koordinaten
    error_x = (min_vertex[0] - known_x) ** 2
    error_y = (min_vertex[1] - known_y) ** 2

    return error_x + error_y + intersection_errors


def find_cube_rotation(known_x, known_y, cut_points):
    """Findet die optimalen Rotationswinkel für den Würfel."""
    initial_guess = [0, 0, 0] # Startwerte für die Winkel
    result = minimize(objective_function, initial_guess, args=(known_x, known_y, cut_points), method='Nelder-Mead')
    return result.x  # Optimierte Winkel


# Beispielwerte für die bekannte Position der unteren Ecke
known_x, known_y = 0.2, -0.3  # Gegebene (x, y)-Koordinaten der unteren Ecke
cut_points = [
    [0.1, -0.2],  # Schnittpunkt der ersten Kante
    [0.3, 0.0],  # Schnittpunkt der zweiten Kante
    [-0.1, 0.2]  # Schnittpunkt der dritten Kante
]

# Berechne die Rotationswinkel
alpha, beta, gamma = find_cube_rotation(known_x, known_y, cut_points)
alpha, beta, gamma = math.degrees(alpha), math.degrees(beta), math.degrees(gamma)
print(f"Geschätzte Rotationswinkel (in Degrees): α={alpha}, β={beta}, γ={gamma}")