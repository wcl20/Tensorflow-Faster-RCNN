import os

BASE_PATH = "lisa"

# Path to annotation file
ANNOTATION_PATH = os.path.sep.join([BASE_PATH, "allAnnotations.csv"])

TRAIN_RECORD = os.path.sep.join([BASE_PATH, "records", "train.record"])
TEST_RECORD = os.path.sep.join([BASE_PATH, "records", "test.record"])

CLASSES = {
    "pedestrianCrossing": 1,
    "signalAhead": 2,
    "stop": 3
}
CLASSES_FILE = os.path.sep.join([BASE_PATH, "records", "classes.pbtxt"])

TEST_SIZE = 0.25
