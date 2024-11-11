import numpy as np
import copy
import cv2
from skimage.segmentation import slic
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from scipy.linalg import eigh
from skimage.util import img_as_float

def SPFCA(image, n_segment, compactness, target=None):
    """
    Superpixel-based principal feature clustering annotation (SPFCA) algorithm.
    refer to: https://doi.org/10.1016/j.matchar.2024.114523
    author: S.L. Lin
    paper title: Superpixel-based principal feature clustering annotation method for dual-phase microstructure segmentation
    
    Parameters:
    - image: Input image which can be a file path or a numpy array.
    - n_segment: The number of superpixels to generate.
    - compactness: Balances color proximity and space proximity. 
                   Higher values give more weight to space proximity, making superpixels more square/cubic.
    - target: Ground truth for accuracy calculation (optional).

    Returns:
    - label: Annotated image with superpixel-based segmentation.
    - accuracy: Accuracy of the segmentation against the target (if provided).
    """

    #### Type inspection and preprocessing ####
    # Check if the image is a file path and read the image
    if type(image) == str:
        image = cv2.imread(image)
        if image is None:
            raise ValueError("Image not found")
    # Ensure the image is a numpy array
    if isinstance(image, np.ndarray) == False:
        image = np.array(image, dtype=np.uint8)
    # Convert color image to grayscale
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Convert the image to a floating-point number and normalize it
    image = img_as_float(image)

    #### Using SLIC algorithm for superpixel segmentation ####
    # Segment the image using the SLIC algorithm
    segments_slic = slic(image, n_segments=n_segment, compactness=compactness, channel_axis=None, start_label=0)
    unique_elements, counts = np.unique(segments_slic, return_counts=True)

    # Create a superpixel sample matrix with unified dimensions
    max_count_index = np.argmax(counts)
    max_counts = counts[max_count_index]
    superpixel_index_list = list(set(segments_slic.flatten()))
    superpixel_samples = np.zeros((len(unique_elements), max_counts), dtype=np.float32)  # Unified dimensional superpixel samples
    mean_value_list = np.zeros(len(unique_elements))  # List of pixel mean values in superpixels

    # Perform superpixel sample sampling
    for superpixel_index in superpixel_index_list:
        mask = (segments_slic != superpixel_index)
        superpixel_matrix = copy.deepcopy(image)
        superpixel_matrix[mask] = 0  # Remove pixels other than the current superpixel
        non_zero_indices = np.nonzero(superpixel_matrix.flatten())  # Find the index of non-zero elements
        if len(non_zero_indices[0]) > 0:
            non_zero_elements = superpixel_matrix.flatten()[non_zero_indices]
            mean_value_list[superpixel_index] = np.mean(non_zero_elements)  # Calculate the mean of pixels within a superpixel
            superpixel_samples[superpixel_index][:len(non_zero_elements)] = non_zero_elements  # Fill the feature values into the superpixel sample matrix
        else:
            mean_value_list[superpixel_index] = 0



    #### Perform principal feature extraction on the superpixel sample matrix ####
    pca = PCA(n_components=15)
    pca.fit(superpixel_samples)
    superpixel_samples = pca.fit_transform(superpixel_samples)

    #### spectralClustering ####
    # Apply spectral clustering to the reduced feature space
    model = SpectralClustering(n_clusters=2, affinity='rbf', random_state=0)
    model.fit(superpixel_samples)

    # annotation
    superpixel_labels = model.fit_predict(superpixel_samples)
    label = np.zeros_like(image)
    mean_value_of_label0 = np.sum(mean_value_list[superpixel_labels == 0])
    mean_value_of_label1 = np.sum(mean_value_list[superpixel_labels == 1])
    if mean_value_of_label1 > mean_value_of_label0:
        superpixel_labels = 1 - superpixel_labels
    for index in range(len(superpixel_labels)):
        if superpixel_labels[index] != 0:
            mask = (segments_slic == index)
            label[mask] = 1
    accuracy = 0
    if target is not None:
        # Calculate accuracy
        accuracy = np.sum(np.logical_and(label, target)) / np.sum(np.logical_or(label, target))
    return label, accuracy

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    image_path = f"./datasets/dataset1/train_images/{3}.jpg"
    # label_path = f"./datasets/dataset1/train_label/{3}.npy"
    # target = np.load(label_path)
    # 读取图片
    oriimage = cv2.imread(image_path)
    # 将彩色图片转换为灰度图片
    gray_image = cv2.cvtColor(oriimage, cv2.COLOR_RGB2GRAY)
    # 将图像转换为浮点数
    image = img_as_float(gray_image)
    label,accuracy = SPFCA(image,n_segment=1000,compactness=0.5,target=None)
    plt.imshow(label)

  