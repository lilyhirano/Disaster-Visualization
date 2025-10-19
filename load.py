## Use this to load


def run(disaster_list):
    import json
    from starter_code.data_utils import get_images, get_labels

    data = {}
    split = "train"
    with open('config.json') as config_file:
        config = json.load(config_file)
        data_dir = config['data_dir']

    for disaster in disaster_list:
        print(f"Loading {split} images and labels for {disaster} dataset...")
        images = get_images(data_dir, disaster, split=split)
        labels = get_labels(data_dir, disaster, split=split)
        data[disaster] = {"images": images, "labels": labels}
    return data