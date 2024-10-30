"""
Programming Assignment 3
Submission Functions
"""

# import packages here

import numpy as np


"""
Q2.1 Eight Point Algorithm
   [I] pts1 -- points in image 1 (Nx2 matrix)
       pts2 -- points in image 2 (Nx2 matrix)
       M -- scalar value computed as max(H, W)
   [O] F -- the fundamental matrix (3x3 matrix)
"""
import numpy as np

def eight_point(pts1, pts2):
    # Import the necessary functions here to avoid circular dependency
    from helper import normalize_points, refineF
    
    pts1, T1 = normalize_points(pts1)
    pts2, T2 = normalize_points(pts2)
    
    N = pts1.shape[0]
    A = np.zeros((N, 9))
    for i in range(N):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    
    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0  # Enforce rank-2
    F = U @ np.diag(S) @ Vt
    
    F = refineF(F, pts1, pts2)
    F = T2.T @ F @ T1
    
    return F

def epipolar_correspondences(im1, im2, F, pts1, window_size=5):
    """
    Compute corresponding points in image 2 for points in image 1 using epipolar geometry.
    """
    pts2 = []
    h, w = im2.shape[:2]
    half_w = window_size // 2
    
    for pt1 in pts1:
        pt1_h = np.array([pt1[0], pt1[1], 1])  # Convert to homogeneous coordinates
        epiline = F @ pt1_h  # Epipolar line in the second image
        epiline /= np.sqrt(epiline[0]**2 + epiline[1]**2)  # Normalize the epipolar line
        
        best_pt2 = None
        min_distance = float('inf')
        
        # Loop over all possible x-coordinates in the second image
        for x2 in range(half_w, w - half_w):
            y2 = int(-(epiline[0] * x2 + epiline[2]) / epiline[1])  # Compute y2 from the epipolar line equation
            
            if half_w <= y2 < h - half_w:
                # Convert pt1 coordinates to integers for slicing
                x1_int, y1_int = int(pt1[0]), int(pt1[1])
                
                # Extract the window around pt1 in image 1
                window1 = im1[y1_int - half_w:y1_int + half_w + 1, x1_int - half_w:x1_int + half_w + 1]
                
                # Extract the window around the corresponding point in image 2
                window2 = im2[y2 - half_w:y2 + half_w + 1, x2 - half_w:x2 + half_w + 1]
                
                # Compute the squared difference between the two windows
                dist = np.sum((window1 - window2) ** 2)
                
                # Track the point with the minimum difference (best match)
                if dist < min_distance:
                    min_distance = dist
                    best_pt2 = [x2, y2]
        
        pts2.append(best_pt2)
    
    return np.array(pts2)


def essential_matrix(F, K1, K2):
    """
    Compute the essential matrix from the fundamental matrix and camera intrinsics.

    Parameters:
    F  -- Fundamental matrix (3x3)
    K1 -- Intrinsic camera matrix of the first camera (3x3)
    K2 -- Intrinsic camera matrix of the second camera (3x3)

    Returns:
    E  -- Essential matrix (3x3)
    """
    # Compute the essential matrix using the formula E = K2.T @ F @ K1
    E = K2.T @ F @ K1

    # Normalize the essential matrix to ensure the last singular value is 1
    U, S, Vt = np.linalg.svd(E)
    S = [1, 1, 0]  # Enforce the condition that the essential matrix has two singular values of 1 and one of 0
    E = U @ np.diag(S) @ Vt

    return E

def triangulate(P1, pts1, P2, pts2):
    N = pts1.shape[0]
    pts3d = []
    
    for i in range(N):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        
        A = np.array([
            y1 * P1[2] - P1[1],
            P1[0] - x1 * P1[2],
            y2 * P2[2] - P2[1],
            P2[0] - x2 * P2[2]
        ])
        
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        pts3d.append(X[:3] / X[3])
    
    return np.array(pts3d)

def rectify_pair(K1, K2, R1, R2, t1, t2):
    c1 = -np.linalg.inv(R1) @ t1
    c2 = -np.linalg.inv(R2) @ t2
    
    r1 = (c1 - c2) / np.linalg.norm(c1 - c2)
    r2 = np.cross(R1[2], r1)
    r3 = np.cross(r1, r2)
    
    R_prime = np.vstack((r1, r2, r3))
    
    R1p = R2p = R_prime
    K1p = K2p = K2
    
    t1p = -R_prime @ c1
    t2p = -R_prime @ c2
    
    M1 = (K1p @ R1p) @ np.linalg.inv(K1 @ R1)
    M2 = (K2p @ R2p) @ np.linalg.inv(K2 @ R2)
    
    return M1, M2, K1p, K2p, R1p, R2p, t1p, t2p

def get_disparity(im1, im2, max_disp, win_size):
    h, w = im1.shape
    disparity_map = np.zeros((h, w))
    half_w = win_size // 2
    
    for y in range(half_w, h - half_w):
        for x in range(half_w, w - half_w):
            min_ssd = float('inf')
            best_disp = 0
            for d in range(max_disp):
                if x - d < half_w:
                    continue
                window1 = im1[y-half_w:y+half_w+1, x-half_w:x+half_w+1]
                window2 = im2[y-half_w:y+half_w+1, x-d-half_w:x-d+half_w+1]
                ssd = np.sum((window1 - window2)**2)
                
                if ssd < min_ssd:
                    min_ssd = ssd
                    best_disp = d
            
            disparity_map[y, x] = best_disp
    return disparity_map

def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    baseline = np.linalg.norm(t1 - t2)
    focal_length = K1[0, 0]
    depth_map = baseline * focal_length / (dispM + 1e-6)
    depth_map[dispM == 0] = 0
    return depth_map




def estimate_pose(x, X):
    N = x.shape[0]  # Number of points
    
    A = []
    for i in range(N):
        X_h = np.append(X[i], 1)  # Convert 3D point to homogeneous coordinates
        x_h = x[i]  # 2D point
        
        # Construct the two rows for each correspondence
        A.append([0, 0, 0, 0, -X_h[0], -X_h[1], -X_h[2], -1, x_h[1]*X_h[0], x_h[1]*X_h[1], x_h[1]*X_h[2], x_h[1]])
        A.append([X_h[0], X_h[1], X_h[2], 1, 0, 0, 0, 0, -x_h[0]*X_h[0], -x_h[0]*X_h[1], -x_h[0]*X_h[2], -x_h[0]])
    
    A = np.array(A)
    
    # Solve using SVD (singular value decomposition)
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)  # The last row of Vt gives us the solution
    
    return P



"""
Q4.2 Camera Parameter Estimation
   [I] P -- camera matrix (3x4 matrix)
   [O] K -- camera intrinsics (3x3 matrix)
       R -- camera extrinsics rotation (3x3 matrix)
       t -- camera extrinsics translation (3x1 matrix)
"""

from scipy.linalg import rq

def estimate_params(P):
    # Extract M matrix (KR) from P matrix
    M = P[:, :3]
    
    # Perform RQ decomposition to get K and R
    K, R = rq(M)
    
    # Ensure the diagonal of K is positive, and adjust R accordingly
    T = np.diag(np.sign(np.diag(K)))
    K = K @ T
    R = T @ R
    
    # Make sure the determinant of R is positive
    if np.linalg.det(R) < 0:
        R = -R
    
    # Compute the camera center (homogeneous coordinates)
    c = -np.linalg.inv(M) @ P[:, 3]
    
    # Compute the translation vector t
    t = -R @ c
    
    return K, R, t
