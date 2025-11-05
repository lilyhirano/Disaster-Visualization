

def crop(disaster_list, data):
    
    for d in disaster_list:
        cropped = []
        for image in data[d]["images"]:

            mask = np.any(image > 0, axis=2)

            rows = np.any(mask, axis=1) #create tuple array of mask
            cols = np.any(mask, axis=0)

            top, bottom = np.where(rows)[0][[0, -1]] #apply tuple to image where first and last are T
            left, right = np.where(cols)[0][[0, -1]]

            cropped.append(image[top:bottom + 1, left:right + 1])

        data[d]["images"] = cropped #keeps disasters in same order

    return data