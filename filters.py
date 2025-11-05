#adds specified channel to images in data

import numpy as np

def apply_filter(data, disaster_list, filter_type):
    from starter_code.feature_utils import get_sobel_features, get_gabor_features, generate_gabor_kernel, get_local_binary_pattern

    for d in disaster_list:
        updated_images = []

        for img in data[d]["images"]:
            img_rgb = img[..., :3]
            if filter_type == "sobel":
                edges = get_sobel_features(img_rgb)
            elif filter_type == "binary":
                edges = get_local_binary_pattern(img_rgb)
            elif filter_type == "gabor":
                theta = 0
                sigma = 1.0
                frequency = 0.1

                kernel = generate_gabor_kernel(theta, sigma, frequency)
                edges = get_gabor_features(img_rgb, kernel)
            else:
                return print("Try different filter name")
            
            expanded = np.expand_dims(edges, axis=-1)
            if img.shape[:2] != expanded.shape[:2]:
                raise ValueError(f"Shape mismatched. image {img.shape} vs filter {expanded.shape}")
            
            stacked = np.concatenate((img, expanded), axis=-1)
            updated_images.append(stacked)


        data[d]["images"] = updated_images

    return data
