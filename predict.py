import argparse
import cv2
import imutils
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Path to frozen checkpoint detection graph")
    parser.add_argument("-l", "--labels", required=True, help="Path to labels file")
    parser.add_argument("-i", "--image", required=True, help="Path to input image")
    parser.add_argument("-n", "--num-classes", required=True, type=int, help="Number of classes")
    parser.add_argument("-c", "--confidence", default=0.2, type=float, help="Confidence threshold")
    args = parser.parse_args()


    COLORS = np.random.uniform(0, 255, size=(args.num_classes, 3))

    model = tf.Graph()
    with model.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(args.model, "rb") as file:
            serialized_graph = file.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name="")

    # Load class labels
    label_map = label_map_util.load_labelmap(args.labels)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=args.num_classes, use_display_name=True
    )
    category_idx = label_map_util.create_category_index(categories)

    with model.as_default():
        with tf.Session(graph=model) as sess:
            image_tensor = model.get_tensor_by_name("image_tensor:0")
            boxes_tensor = model.get_tensor_by_name("detection_boxes:0")

            scores_tensor = model.get_tensor_by_name("detection_scores:0")
            classes_tensor = model.get_tensor_by_name("detection_classes:0")
            num_detections = model.get_tensor_by_name("num_detections:0")

            image = cv2.imread(args.image)

            # Resize image
            height, width = image.shape[:2]
            if width > height and width > 1000:
                image = imutils.resize(image, width=1000)
            elif height > width and height > 1000:
                image = imutils.resize(image, height=1000)
            height, width = image.shape[:2]

            # Prepare model input
            output = image.copy()
            image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
            image = np.expand_dims(image, axis=0)

            boxes, scores, labels, N = sess.run(
                [boxes_tensor, scores_tensor, classes_tensor, num_detections],
                feed_dict={ image_tensor: image }
            )
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            labels = np.squeeze(labels)

            for box, score, label in zip(boxes, scores, labels):
                print(box, score, label)
                if score < args.confidence:
                    continue
                # Get bounding box coordinates
                y1, x1, y2, x2 = box
                x1 = int(x1 * width)
                y1 = int(y1 * height)
                x2 = int(x2 * width)
                y2 = int(y2 * height)
                # Get label and label index
                label = category_idx[label]
                idx = int(label["id"]) - 1
                label = f"{label['name']}:{score:.2f}"
                # Draw bouding box
                cv2.rectangle(output, (x1, y1), (x2, y2), COLORS[idx], 2)
                # Draw label
                y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                cv2.putText(output, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS[idx], 1)

            cv2.imwrite("output.png", output)

if __name__ == '__main__':
    main()
