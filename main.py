from tqdm import tqdm
from ultralytics import YOLO
import cv2
import numpy as np
from typing import List
import glob

# Load a model
pred_model = YOLO("/Users/nadaveidelstein/Downloads/yolo11n.pt")
embed_model = YOLO("/Users/nadaveidelstein/Downloads/yolo11s-cls.pt")

# pred_model.compile()
# embed_model.compile()

def norm(feats):
    return feats / np.linalg.norm(feats)
    # return feats

def bboxes(image):
    # Predict with the model
    results = pred_model(image)[0]

    def gen():
        # Per object (bounding box) in the results, perform feature-extraction on each object
        for result in results:
            boxes = result.boxes or []
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = image[y1:y2, x1:x2]
                # resize so it's 320x320 with padding
                crop = cv2.resize(crop, (320, 320), interpolation=cv2.INTER_AREA)
                feats = embed_model.embed(crop)[0]
                yield crop, norm(feats)

    return len(results.boxes or []), gen

vec_store: List[tuple[np.ndarray, np.ndarray]] = []

def build_db(paths: List[str]):
    for p in tqdm(paths):
        image = cv2.imread(p)
        num_results, gen = bboxes(image)
        if num_results != 1:
            continue

        print(f"Adding {p} to database")
        for crop, feats in gen():
            vec_store.append((norm(feats), crop))


def lookup_one(query: np.ndarray, feats: List[np.ndarray], crops: List[np.ndarray]):
    distances = np.dot(feats, query)
    closest = np.argmax(distances)
    print("closest:", closest, "distances:", distances, "lenfeats:", len(feats))
    return feats[closest], crops[closest], distances[closest]

def lookup_image(image_path, feats, crops):
    image = cv2.imread(image_path)

    num_results, gen = bboxes(image)
    # if num_results == 0:
        # print(f"Skipping {image_path} because it has {num_results} results")

    for query_crop, query in gen():
        _, result_crop, dist = lookup_one(query, feats, crops)
        yield query_crop, result_crop, dist

def main():
    # db_paths = ["/Users/nadaveidelstein/Downloads/4750.jpg"]
    db_paths = glob.glob("/Users/nadaveidelstein/Downloads/IKEADataset/**/*.jpg", recursive=True)[:1500]
    print(f"Building database with {len(db_paths)} images")

    build_db(db_paths)

    feats = [normed for normed, _ in vec_store]
    crops = [crop for _, crop in vec_store]

    image_path = "/Users/nadaveidelstein/Downloads/generated.jpeg"
    # TODO: make sure these are correct
    for query_bbox, result_bbox, distance in lookup_image(image_path, feats, crops):
        # Resize both images to the larger dimensions so they're concatenable
        cv2.imshow("Query", query_bbox)
        cv2.imshow("Result", result_bbox)
        print(f"Distance: {distance:.2f}")
        cv2.waitKey(0)


main()
