from skimage import filters, color, feature
import numpy as np
from skimage.morphology import disk

X = np.load("all_images.npy")
#Y = np.load("all_labels.npy")


# features = []
# for img in X:
#     img_gray = np.mean(img, axis=2)
#     filtered, i = filters.gabor(img_gray, frequency=0.6)
#     features.append(np.mean(filtered))

# final = np.array(features).reshape(-1, 1)


# np.save("gabor_mean_all", final)



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


# np.save("sobel_mean_all.npy", features1)
# np.save("sobel_var_all.npy", features2)
# np.save("sobel_edges0.4_all.npy", features3)

# features1 = []
# features2 = []
# for img in X:
#     img_gray = color.rgb2gray(img)
#     filtered = feature.local_binary_pattern(img_gray, P=8, R=1, method="uniform")
#     features1.append(np.median(filtered))
#     features2.append(np.mean(filtered))

# features1 = np.array(features1).reshape(-1, 1)
# features2 = np.array(features2).reshape(-1, 1)

# np.save("lbp_median_all.npy", features1)
# np.save("lbp_mean_all.npy", features2)


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
# np.save("entropy_median_all.npy", features1)
# np.save("entropy_mean_all.npy", features2)


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

# np.save("mean_R.npy", features1)
# np.save("mean_G.npy", features2)
# np.save("mean_B.npy", features3)

# feature = []
# for img in X:
#     img_gray = color.rgb2gray(img)
#     feature.append(np.mean(img_gray))

# feature = np.array(feature).reshape(-1, 1)
# np.save("mean_Grey.npy", feature)
