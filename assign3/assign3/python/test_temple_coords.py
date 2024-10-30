import cv2
import numpy as np
import helper as hlp
import submission as sub
import matplotlib.pyplot as plt

"""
Part 1 (Q2): Sparse Reconstruction
"""

def plot_epipolar_lines(im1, im2, F, pts1):
    """
    Manually plot epipolar lines on the second image, given points from the first image.
    """
    plt.subplot(1, 2, 1)
    plt.imshow(im1)
    plt.scatter(pts1[:, 0], pts1[:, 1], c='r', marker='o')
    plt.title("Image 1 with selected points")

    plt.subplot(1, 2, 2)
    plt.imshow(im2)
    
    # For each point in the first image, compute and plot the epipolar line in the second image
    for pt1 in pts1:
        pt1_h = np.append(pt1, 1)  # Convert point to homogeneous coordinates
        epiline = F @ pt1_h  # Compute the epipolar line in the second image

        # Epipolar line parameters (a*x + b*y + c = 0)
        a, b, c = epiline

        # Compute the x coordinates for the start and end of the image
        x_vals = np.array([0, im2.shape[1]])
        y_vals = -(a * x_vals + c) / b  # Solve for y in terms of x

        # Plot the epipolar line
        plt.plot(x_vals, y_vals, 'b')

    plt.title("Image 2 with epipolar lines")
    plt.show()

def main():
    # 1. Load the two temple images and the points from data/some_corresp.npz
    im1 = cv2.imread("../data/im1.png")
    im2 = cv2.imread("../data/im2.png")
    corresp = np.load("../data/some_corresp.npz")
    pts1_corresp = corresp['pts1']
    pts2_corresp = corresp['pts2']

    # OpenCV uses BGR, while matplotlib uses RGB
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

    # 2. Run eight_point to compute F
    F = sub.eight_point(pts1_corresp, pts2_corresp)

    # 3. Manually select points in the first image before calling hlp.displayEpipolarF()
    plt.imshow(im1)
    plt.title("Select points in the first image (Press Enter when done)")
    selected_pts1 = plt.ginput(n=-1, timeout=0)  # Left-click to select points, press Enter to finish
    plt.close()

    if len(selected_pts1) == 0:
        print("No points were selected. Exiting the program.")
        return  # Exit if no points are selected

    # Convert selected points to NumPy array
    selected_pts1 = np.array(selected_pts1)

    # 4. Manually plot epipolar lines on the second image
    plot_epipolar_lines(im1, im2, F, selected_pts1)

    # 5. Run epipolar_correspondences to get corresponding points in image 2
    pts2_temple = sub.epipolar_correspondences(im1, im2, F, selected_pts1)

    # Print selected points and their correspondences
    print("Selected Points in Image 1:", selected_pts1)
    print("Corresponding Points in Image 2:", pts2_temple)

    # 6. Load intrinsics and compute the camera projection matrix P1
    intrinsics = np.load("../data/intrinsics.npz")
    K1 = intrinsics["K1"]
    K2 = intrinsics["K2"]
    R1 = np.eye(3)
    t1 = np.zeros((3, 1))
    P1 = K1 @ np.hstack((R1, t1))  # Compute the projection matrix for the first camera

    # 7. Compute essential matrix
    E = sub.essential_matrix(F, K1, K2)

    # 8. Use camera2 to get 4 camera projection matrices P2
    M2_candidates = hlp.camera2(E)  # Returns 4 possible M2 matrices

    # 9. Run triangulate using the projection matrices
    best_P2 = None
    best_pts3d = None

    for i, M2 in enumerate(M2_candidates):
        print(f"Shape of M2: {M2.shape}")  # Debugging: Print the shape of M2

    # Since M2 is 4x4, extract the first 3 rows and all 4 columns to get the 3x4 matrix
        M2 = M2[:3, :]  # Extract the top 3 rows (ignore the last row)

    # Now M2 should be 3x4, so we can split it into rotation and translation
        R2 = M2[:, :3]  # 3x3 rotation matrix
        t2 = M2[:, 3]   # 3x1 translation vector

    # Reshape t2 to ensure it's a column vector (3x1)
        t2 = t2.reshape(3, 1)

    # Compute the projection matrix P2 using K2, R2, and t2
        P2 = K2 @ np.hstack((R2, t2))  # 3x4 projection matrix

    # Run triangulate using the projection matrices
        pts3d = sub.triangulate(P1, selected_pts1, P2, pts2_temple)

    # Check how many points are in front of both cameras
        points_in_front = np.mean(pts3d[:, 2] > 0)  # Compute the percentage of points with z > 0
        print(f"M2 candidate {i + 1}: {points_in_front * 100:.2f}% of points are in front of the camera")

    # Check if most of the 3D points are in front of both cameras
        if points_in_front > 0.75:  # At least 75% of the points should be in front
            best_P2 = P2
            best_pts3d = pts3d
            print(f"Selected M2 candidate {i + 1} with {points_in_front * 100:.2f}% points in front")
            break

# Ensure the correct P2 is chosen
    if best_P2 is None:
        print("No valid M2 matrix where points are in front of both cameras.")
        raise RuntimeError("No valid P2 matrix found where points are in front of both cameras.")


    # 10. Compute the reprojection error
    reprojection_error1 = hlp.reprojection_error(best_pts3d, selected_pts1, P1)
    reprojection_error2 = hlp.reprojection_error(best_pts3d, pts2_temple, best_P2)
    
    print(f"P1 reprojection error: {reprojection_error1}")
    print(f"P2 reprojection error: {reprojection_error2}")

    # 11. Scatter plot the correct 3D points
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(best_pts3d[:, 0], best_pts3d[:, 2], -best_pts3d[:, 1])
    ax.set_xlim(-1, 1)
    ax.set_ylim(3, 5)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    plt.tight_layout()
    plt.show()

    # 12. Save the computed extrinsic parameters (R1, R2, t1, t2) to data/extrinsics.npz
    R2_t2 = np.linalg.inv(K2) @ best_P2
    R2 = R2_t2[:, :3]  # Rotation matrix
    t2 = R2_t2[:, 3, np.newaxis]  # Translation vector
    np.savez("../data/extrinsics.npz", R1=R1, t1=t1, R2=R2, t2=t2)


if __name__ == "__main__":
    main()
