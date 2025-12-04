import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import cv2
from skimage import color, feature, filters
from skimage.morphology import disk

def extract_selected_features(images, save_path="data/", suffix="classification"):
    """
    Extracts selected features: lbp_mean, entropy_mean, blue, green
    and saves each as a separate .npy file in save_path.
    
    Parameters:
    - images: list or array of RGB images
    - save_path: folder to save .npy files
    - suffix: string to append at the end of each filename
    """
    import os

    os.makedirs(save_path, exist_ok=True)
    
    lbp_mean_list = []
    entropy_mean_list = []
    blue_list = []
    green_list = []
    
    for img in images:
        # Convert to grayscale once for LBP and entropy
        img_gray = color.rgb2gray(img)
        
        # LBP
        lbp = feature.local_binary_pattern(img_gray, P=8, R=1, method="uniform")
        lbp_mean_list.append(np.mean(lbp))
        
        # Entropy
        img_gray_uint = (img_gray * 255).astype(np.uint8)
        entropy_img = filters.rank.entropy(img_gray_uint, disk(3))
        entropy_mean_list.append(np.mean(entropy_img))
        
        # Blue and Green channel means
        blue_list.append(np.mean(img[:,:,2]))
        green_list.append(np.mean(img[:,:,1]))
    
    np.save(save_path + f"lbp_mean_{suffix}.npy", np.array(lbp_mean_list).reshape(-1,1))
    np.save(save_path + f"entropy_mean_{suffix}.npy", np.array(entropy_mean_list).reshape(-1,1))
    np.save(save_path + f"blue_{suffix}.npy", np.array(blue_list).reshape(-1,1))
    np.save(save_path + f"green_{suffix}.npy", np.array(green_list).reshape(-1,1))

images = np.load("./npy_files/2_MERGED_imgs.npy")
extract_selected_features(images)