import numpy as np
from scipy.spatial.transform import Rotation as R


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


def are_cube_rotations_equivalent(euler1, euler2, degrees=True, tol=1e-6):
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


# Example usage:
euler_angle_1 = (20, 90, 0)
euler_angle_2 = (20, 0, 45)  # Should be equivalent

print(are_cube_rotations_equivalent(euler_angle_1, euler_angle_2))