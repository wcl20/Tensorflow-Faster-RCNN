import io
import os
import tensorflow as tf
import tqdm
from config import config
from PIL import Image
from object_detection.utils import dataset_util
from sklearn.model_selection import train_test_split

def main(_):

    # Write classes file
    os.makedirs(os.path.dirname(config.CLASSES_FILE), exist_ok=True)
    file = open(config.CLASSES_FILE, "w")
    for label, id in config.CLASSES.items():
        file.write("item {\n\tid: " + str(id) + "\n\tname: '" + label + "'\n}\n")
    file.close()

    annotations = {}
    # Parse annotation file
    rows = open(config.ANNOTATION_PATH).read().strip().split("\n")
    # Ignore first line header
    for row in rows[1:]:
        # Get image path and bounding box coordinates
        row, _ = row.split(",")
        img_path, label, x1, y1, x2, y2, _ = row.split(";")
        if label not in config.CLASSES:
            continue

        img_path = os.path.sep.join([config.BASE_PATH, img_path])
        x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])

        # Add image path and bounding box information to dictionary
        info = annotations.get(img_path, [])
        info.append((label, (x1, y1, x2, y2)))
        annotations[img_path] = info

    train_paths, test_paths = train_test_split(list(annotations.keys()), test_size=config.TEST_SIZE, random_state=42)

    datasets = [
        ("train", train_paths, config.TRAIN_RECORD),
        ("test", test_paths, config.TEST_RECORD)
    ]

    for type, img_paths, output_path in datasets:
        print(f"[INFO] Building {output_path} ...")
        writer = tf.python_io.TFRecordWriter(output_path)

        for img_path in tqdm.tqdm(img_paths):
            # Load image as tf object
            encoded = tf.gfile.GFile(img_path, "rb").read()

            # Get image size
            image = Image.open(io.BytesIO(encoded))
            width, height = image.size[:2]

            # Get filename and format
            filename = img_path.split(os.path.sep)[-1]
            img_format = filename[filename.rfind(".") + 1:]

            # Get bounding boxes
            xmins, xmaxs = [], []
            ymins, ymaxs = [], []
            classes_texts, classes = [], []
            for label, (x1, y1, x2, y2) in annotations[img_path]:
                xmins.append(x1 / width)
                xmaxs.append(x2 / width)
                ymins.append(y1 / height)
                ymaxs.append(y2 / height)
                classes_texts.append(label.encode("utf8"))
                classes.append(config.CLASSES[label])

            features = tf.train.Features(feature = {
                "image/height": dataset_util.int64_feature(height),
                "image/width": dataset_util.int64_feature(width),
                "image/filename": dataset_util.bytes_feature(filename.encode("utf8")),
                "image/source_id": dataset_util.bytes_feature(filename.encode("utf8")),
                'image/encoded': dataset_util.bytes_feature(encoded),
                "image/format": dataset_util.bytes_feature(img_format.encode("utf8")),
                "image/object/bbox/xmin": dataset_util.float_list_feature(xmins),
                "image/object/bbox/xmax": dataset_util.float_list_feature(xmaxs),
                "image/object/bbox/ymin": dataset_util.float_list_feature(ymins),
                "image/object/bbox/ymax": dataset_util.float_list_feature(ymaxs),
                "image/object/class/text": dataset_util.bytes_list_feature(classes_texts),
                "image/object/class/label": dataset_util.int64_list_feature(classes)
            })
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())
        writer.close()

if __name__ == '__main__':
    tf.app.run()
