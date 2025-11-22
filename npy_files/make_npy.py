from skimage import filters, color, feature
import numpy as np
from skimage.morphology import disk

X = np.load("npy_files/2_MERGED_imgs.npy")
#X = np.load("npy_files/hurricane-matthew_imgs.npy")
#Y = np.load("all_labels.npy")


# features = []
# for img in X:
#     img_gray = np.mean(img, axis=2)
#     filtered, i = filters.gabor(img_gray, frequency=0.6)
#     features.append(np.mean(filtered))

# final = np.array(features).reshape(-1, 1)


# np.save("gabor_mean_2", final)



# features1 = []
# features2 = []
# features3 = []

# for img in X:
#     img_gray = color.rgb2gray(img)
#     filtered = filters.sobel(img_gray)
#     features1.append(np.mean(filtered))
#     features2.append(np.var(filtered))
#     filtered_norm = filtered / filtered.max()
#     features3.append(np.mean(filtered_norm > 0.4))

# features1 = np.array(features1).reshape(-1, 1)
# features2 = np.array(features2).reshape(-1, 1)
# features3= np.array(features3).reshape(-1, 1)


# np.save("sobel_mean_2.npy", features1)
# np.save("sobel_var_2.npy", features2)
# np.save("sobel_edges0.4_2.npy", features3)

# features1 = []
# features2 = []
# for img in X:
#     img_gray = color.rgb2gray(img)
#     filtered = feature.local_binary_pattern(img_gray, P=8, R=1, method="uniform")
#     features1.append(np.median(filtered))
#     features2.append(np.mean(filtered))

# features1 = np.array(features1).reshape(-1, 1)
# features2 = np.array(features2).reshape(-1, 1)

# np.save("lbp_median_2.npy", features1)
# np.save("lbp_mean_2.npy", features2)


# features1 = []
# features2 = []
# for img in X:
#     img_gray = color.rgb2gray(img)
#     img_gray = (img_gray * 255).astype(np.uint8)
#     filtered = filters.rank.entropy(img_gray, disk(3))
#     features1.append(np.median(filtered))
#     features2.append(np.mean(filtered))


# features1 = np.array(features1).reshape(-1, 1)
# features2 = np.array(features2).reshape(-1, 1)
# np.save("entropy_median_2.npy", features1)
# np.save("entropy_mean_2.npy", features2)


# features1 = []
# features2 = []
# features3 = []

# for img in X:
#     mean_rgb = np.mean(img, axis=(0, 1))  

#     features1.append(mean_rgb[0]) #red
#     features2.append(mean_rgb[1]) #green
#     features3.append(mean_rgb[2]) #blue



# features1 = np.array(features1).reshape(-1, 1)
# features2 = np.array(features2).reshape(-1, 1)
# features3= np.array(features3).reshape(-1, 1)

# np.save("mean_R_2.npy", features1)
# np.save("mean_G_2.npy", features2)
# np.save("mean_B_2.npy", features3)

# feature = []
# for img in X:
#     img_gray = color.rgb2gray(img)
#     feature.append(np.mean(img_gray))

# feature = np.array(feature).reshape(-1, 1)
# np.save("mean_Grey_2.npy", feature)


feature = []
for img in X:
    mean_color = np.mean(img)
    feature.append(mean_color)

feature = np.array(feature).reshape(-1, 1)
np.save("mean_color_2.npy", feature)

