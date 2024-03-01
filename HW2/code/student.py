import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops
import cv2
import scipy as sp

"""
Next steps:
- Gaussian filter for the 16x16 grid
- Rotation invariant by subtracting all directions by center point direction
- Multiscale interest point detection
"""



def plot_interest_points(image, x, y):
    '''
    Plot interest points for the input image. 
    
    Show the interest points given on the input image. Be sure to add the images you make to your writeup. 

    Useful functions: Some helpful (not necessarily required) functions may include
        - matplotlib.pyplot.imshow, matplotlib.pyplot.scatter, matplotlib.pyplot.show, matplotlib.pyplot.savefig
    
    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    '''

    # TODO: Your implementation here! See block comments and the handout pdf for instructions

    plt.imshow(image, cmap='gray')
    plt.scatter(x, y, c='r', marker='o')
    plt.show()



def get_interest_points(image, feature_width):
    '''
    Returns interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 7.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops
          
    Note: You may decide it is unnecessary to use feature_width in get_interest_points, or you may also decide to 
    use this parameter to exclude the points near image edges.

    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width: the width and height of each local feature in pixels

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    '''

    # TODO: Your implementation here! See block comments and the handout pdf for instructions

    # These are placeholders - replace with the coordinates of your interest points!

    xs = np.zeros(1)
    ys = np.zeros(1)
    # Note that xs and ys represent the coordinates of the image. Thus, xs actually denote the columns
    # of the respective points and ys denote the rows of the respective points.

    # STEP 1: Calculate the gradient (partial derivatives on two directions).
    # optionally blur the image first
    blurred_image = filters.gaussian(image, sigma=1)
    Ix = filters.sobel_v(blurred_image)
    Iy = filters.sobel_h(blurred_image)

    Ix2 = Ix**2
    Iy2 = Iy**2
    IxIy = Ix * Iy

    # STEP 2: Apply Gaussian filter with appropriate sigma.
    sigma = 1
    blurred_Ix2 = filters.gaussian(Ix2, sigma=sigma)
    blurred_Iy2 = filters.gaussian(Iy2, sigma=sigma)
    blurred_IxIy = filters.gaussian(IxIy, sigma=sigma)

    # STEP 3: Calculate Harris cornerness score for all pixels. 
    alpha = 0.05
    C = (blurred_Ix2 * blurred_Iy2 - blurred_IxIy**2) - alpha * (blurred_Ix2 + blurred_Iy2)**2
    # C_truncated = C[feature_width:-feature_width, feature_width:-feature_width]
    C_truncated = C

    # STEP 4: Peak local max to eliminate clusters. (Try different parameters.)
    threshold = np.percentile(C_truncated, 90)
    min_distance = 20
    coordinates = feature.peak_local_max(C_truncated, min_distance=min_distance, threshold_abs=threshold, exclude_border=feature_width)
    # coordinates = feature.peak_local_max(C_truncated, min_distance=min_distance, threshold_abs=threshold)
    # print("num coordinates: ", len(coordinates))
    # xs = coordinates[:, 1] + feature_width
    # ys = coordinates[:, 0] + feature_width
    xs = coordinates[:, 1] 
    ys = coordinates[:, 0]
    
    # BONUS: There are some ways to improve:
    # 1. Making interest point detection multi-scaled.
    # 2. Use adaptive non-maximum suppression.

    return xs, ys


def get_features(image, x, y, feature_width):
    '''
    Returns features for a given set of interest points.

    To start with, you might want to simply use normalized patches as your
    local feature descriptor. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like feature descriptor
    (See Szeliski 7.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) feature descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like features can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments. Make sure input arguments 
    are optional or the autograder will break.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    '''

    # TODO: Your implementation here! See block comments and the handout pdf for instructions
    # print(f"Getting features for {len(x)} interest points.")


    def getOrientedFilterUsingDistribution(degree, sigma):

        degree = degree + np.pi/2

        # generate the coordinates
        x,y = np.mgrid[-1:2, -1:2]

        # rotate the coordinates
        x_rot = x * np.cos(degree) - y * np.sin(degree)
        y_rot = x * np.sin(degree) + y * np.cos(degree)

        # create the kernel
        kernel = np.zeros((3, 3))

        # create the multivariate normal distribution
        mvn = sp.stats.multivariate_normal([0, 0], [[sigma**2, 0], [0, sigma**2]])

        # sample the distribution at each coordinate
        kernel = -x_rot * mvn.pdf(np.stack([x_rot, y_rot], axis=-1))

        # normalize the kernel
        kernel /= np.sum(np.abs(kernel))

        return kernel

    def orientedFilterMagnitude(im: np.ndarray, filter_type_dist: bool = True):
    # return mag: np.ndarray, theta: np.ndarray
    # orientations would be 0, 45, 90, 135 degrees

        sigma = 1
        im = cv2.GaussianBlur(im, (5, 5), sigma)
        # im = cv2.normalize(im, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32FC1) // this makes it worse, but for the previous part, this made it better, ASK
        
        num_rows = len(im)
        num_cols = len(im[0])

        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
        gray_mags = []
        magnitude_array = []
        
        final_mags = np.ones((num_rows, num_cols))

        filter_set = []
        
        for orientation in orientations:
            # a kernel size of 3x3 was assumed

            orientation = -1 * orientation
            
            filter_dist = getOrientedFilterUsingDistribution(orientation, sigma=1)

            # apply the filter
            gray_mags = cv2.filter2D(im, -1, filter_dist)
            magnitude_array.append(gray_mags)

            filter_set.append(filter)

        magnitude_array = np.array(magnitude_array)
        max_magnitude_index = np.argmax(magnitude_array, axis=0)
        final_mags = np.take_along_axis(magnitude_array, max_magnitude_index[None, ...], axis=0)[0]
        final_thetas = np.array(orientations)[max_magnitude_index]

        return final_mags, final_thetas, filter_set


    # print(image.dtype)

    # image_blurred = filters.gaussian(image, sigma=1)
    
    # # # STEP 1: Calculate the gradient (partial derivatives on two directions) on all pixels.
    # dx = filters.sobel_v(image_blurred)
    # dy = filters.sobel_h(image_blurred)

    # # # STEP 2: Decompose the gradient vectors to magnitude and direction.
    # magnitude = np.sqrt(dx**2 + dy**2)
    # direction = np.arctan2(dy, dx)

    # method 0, using the oriented filter
    # magnitude, direction, filter_set = orientedFilterMagnitude(image)

    # STEP 3: For each interest point, calculate the local histogram based on related 4x4 grid cells.
    #         Each cell is a square with feature_width / 4 pixels length of side.
    #         For each cell, we assign these gradient vectors corresponding to these pixels to 8 bins
    #         based on the direction (angle) of the gradient vectors. 

    # Method 1: gradient calculation
    image_blurred = cv2.GaussianBlur(image, (5, 5), 1)

    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = sobel_x.T

    x_gradient = cv2.filter2D(image_blurred, -1, sobel_x)
    y_gradient = cv2.filter2D(image_blurred, -1, sobel_y)
    direction = np.arctan2(y_gradient, x_gradient)
    direction = (direction + 2*np.pi) % (2*np.pi)
    magnitude = np.sqrt(x_gradient**2 + y_gradient**2)

    # print(f"direction max: {np.max(direction)}")
    # print(f"direction min: {np.min(direction)}")

    # Method 2: gradient calculation using skimage, this performs worse for some reason
    # image_blurred = filters.gaussian(image, sigma=1)
    # x_gradient = filters.sobel_v(image_blurred)
    # y_gradient = filters.sobel_h(image_blurred)
    # direction = np.arctan2(y_gradient, x_gradient)
    # magnitude = np.sqrt(x_gradient**2 + y_gradient**2)

    # print(f"all X: {x}")
    # print(f"all Y: {y}")

    # plt.imshow(magnitude, cmap='gray')

    features = []

    def normalize_array(arr):
        normalized_arr = arr / np.linalg.norm(arr) if np.linalg.norm(arr) != 0 else arr
        return normalized_arr

    # dirs_considered = []
    # indices = []

    # Iterate over keypoints
    for y_coordinate, x_coordinate in zip(x, y):

        x_coordinate = int(x_coordinate)
        y_coordinate = int(y_coordinate)

        # get the gradient vector for the center pixel
        # center_pixel_direction = direction[x_coordinate][y_coordinate] # this shouldn't be breaking, we aren't out of the image

        # Compute histogram of gradients
        histogram16x16 = []
        for i in range(-2,2):
            for j in range(-2,2):
                # Compute histogram of the patch
                histogram4x4 = np.zeros(8)
                for k in range(4):
                    for l in range(4):
                        original_x = x_coordinate + (i * 4) + k
                        original_y = y_coordinate + (j * 4) + l
                        
                        # Check if the pixel is out of the image
                        if original_x < 0 or original_x >= image.shape[0] or original_y < 0 or original_y >= image.shape[1]:
                            continue

                        # histogram4x4[int(direction[original_x][original_y] / (np.pi / 4))] += magnitude[original_x, original_y]
                        histogram4x4[int((direction[original_x][original_y]) / (np.pi / 4))] += 1         # more accuracy as of now
                        # dirs_considered.append(direction[original_x][original_y])
                        # indices.append(int(direction[original_x][original_y] / (np.pi / 4)))
                        
                        # histogram4x4[int(direction[original_x][original_y] / (np.pi / 4))] += 1
                # histogram4x4 /= np.linalg.norm(histogram4x4)
                # histogram4x4 /= np.max(histogram4x4)
                histogram4x4 = normalize_array(histogram4x4)
                histogram16x16.append(histogram4x4)
        histogram16x16 = np.array(histogram16x16).flatten()
        histogram16x16 = normalize_array(histogram16x16)
        features.append(histogram16x16)

    # print(f"dirs considered max: {np.max(dirs_considered)}")
    # print(f"dirs considered min: {np.min(dirs_considered)}")
    # print(f"indices max: {np.max(indices)}")
    # print(f"indices min: {np.min(indices)}")

    # STEP 4: Now for each cell, we have a 8-dimensional vector. Appending the vectors in the 4x4 cells,
    #         we have a 128-dimensional feature.
    # print("features shape: ", np.array(features).shape)
    


    # STEP 5: Don't forget to normalize your feature.
    
    # BONUS: There are some ways to improve:
    # 1. Use a multi-scaled feature descriptor.
    # 2. Borrow ideas from GLOH or other type of feature descriptors.

    # This is a placeholder - replace this with your features!
    # features = np.zeros((len(x),128))
    
    # [(613.3341121495325, 72.62383177570086)]
    # [(541.2344479004664, 142.65590979782246)]

    # find index of 613.3341121495325 in x
    # idx_x1 = np.where(x == 613.3341121495325)
    # print(f"idx_x1: {idx_x1}")

    # idx_x2 = np.where(x == 541.2344479004664)
    # print(f"idx_x2: {idx_x2}")

    # print(f"at 3: {x[3]}, {y[3]}")
    # print(list(features[3]))

    
    return features


def match_features(im1_features, im2_features):
    '''
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 7.18 in Section 7.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    '''

    # TODO: Your implementation here! See block comments and the handout pdf for instructions

    # These are placeholders - replace with your matches and confidences!
    
    # STEP 1: Calculate the distances between each pairs of features between im1_features and im2_features.
    #         HINT: check match_features.pdf
    # convert im1_features and im2_features to numpy arrays
    f1 = np.array(im1_features)
    f2 = np.array(im2_features)
    f1_squared = np.sum(np.square(f1), axis=1).reshape(-1, 1)
    f2_squared = np.sum(np.square(f2), axis=1).reshape(-1, 1)
    A = f1_squared + f2_squared.T
    B = -2 * np.dot(f1, f2.T)
    D = np.sqrt(A + B)
    # STEP 2: Sort and find closest features for each feature, then performs NNDR test.

    threshold = 0.95

    # ratios = []

    matches = []
    confidences = []
    for i in range(D.shape[0]):
        sorted_indices = np.argsort(D[i])
        ratio = D[i][sorted_indices[0]] / D[i][sorted_indices[1]] if D[i][sorted_indices[1]] != 0 else 0
        # ratios.append(ratio)
        if ratio < threshold:
            matches.append([i, sorted_indices[0]])
            confidences.append(1 - D[i][sorted_indices[0]] / D[i][sorted_indices[1]])
    matches = np.array(matches)
    confidences = np.array(confidences)
    
    # print("ratios: ", ratios)

    # BONUS: Using PCA might help the speed (but maybe not the accuracy).

    # matches = np.zeros((1,2))
    # confidences = np.zeros(1)


    return matches, confidences
