import numpy as np
import matplotlib.pyplot as plt
from submission import estimate_pose, estimate_params

def project_cad():
    data = np.load('../data/pnp.npz')
    X, x, cad = data['X'], data['x'], data['cad']
    
    P = estimate_pose(x, X)
    K, R, t = estimate_params(P)
    
    # Project 3D CAD model onto image
    cad_projected = (K @ (R @ cad.T + t.reshape(-1, 1))).T
    cad_projected = cad_projected[:, :2] / cad_projected[:, 2:3]
    
    plt.scatter(cad_projected[:, 0], cad_projected[:, 1], c='blue', label='CAD Projection')
    plt.scatter(x[:, 0], x[:, 1], c='red', label='Original Points')
    plt.legend()
    plt.show()

project_cad()
