import os
import cv2
import mediapipe as mp
import numpy as np
import joblib
import json
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

DATASET_PATH = "dataset_ang"
OUTPUT_NPZ = "ang_dataset.npz"
MODEL_FILE = "ang_svm_model.pkl"
LABEL_MAP_FILE = "label_map_ang.json"
MIN_SAMPLES_PER_CLASS = 2

class AngFeatureExtractor:

    def __init__(self, dataset_path=DATASET_PATH, max_num_hands=1):
        self.dataset_path = dataset_path
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=max_num_hands,
            min_detection_confidence=0.7
        )

    def _extract_from_landmarks(self, hand_landmarks):

        coords = []
        for lm in hand_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        return np.array(coords, dtype=np.float32)

    def load_dataset(self):

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset folder '{self.dataset_path}' not found.")

        X_list = []
        y_list = []

        folders = sorted(os.listdir(self.dataset_path))
        if not folders:
            raise RuntimeError(f"No subfolders found in '{self.dataset_path}'.")

        print(f"Found label folders: {folders}\n")

        for label in folders:
            folder_path = os.path.join(self.dataset_path, label)
            if not os.path.isdir(folder_path):
                continue

            print(f"[INFO] Processing label '{label}' ({folder_path})")
            for fname in tqdm(sorted(os.listdir(folder_path))):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                img_path = os.path.join(folder_path, fname)
                img = cv2.imread(img_path)
                if img is None:
                    continue


                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.hands.process(img_rgb)


                if not results.multi_hand_landmarks:
                    continue


                hand_landmarks = results.multi_hand_landmarks[0]
                features = self._extract_from_landmarks(hand_landmarks)

                X_list.append(features)
                y_list.append(label)

        if not X_list:
            raise RuntimeError("No hand landmarks were extracted from any images")

        X = np.vstack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=object)

        print(f"\nExtraction complete: {X.shape[0]} samples, {X.shape[1]} features.")
        return X, y


def save_npz(X, y, path=OUTPUT_NPZ):
    np.savez(path, X=X, y=y)
    print(f"[INFO] Saved dataset to {path}")


def train_and_save_svm(X, y):

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    index_to_label = {int(idx): label for idx, label in enumerate(le.classes_)}
    with open(LABEL_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(index_to_label, f, indent=4, ensure_ascii=False)
    print(f"Saved label map to {LABEL_MAP_FILE}")

    unique, counts = np.unique(y_encoded, return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))
    print("Class distribution (encoded_label:count):", dist)

    test_size = 0.2
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
    except ValueError as e:

        print("Stratified split failed:", str(e))
        print("Falling back to non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, shuffle=True
        )

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True))
    ])

    print("Training SVM")
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc*100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    joblib.dump({"pipeline": pipeline, "label_encoder": le}, MODEL_FILE)
    print(f"Saved model and label encoder to {MODEL_FILE}")


def main():
    print("[INFO] Starting feature extraction from dataset_ang...")
    extractor = AngFeatureExtractor(dataset_path=DATASET_PATH)
    X, y = extractor.load_dataset()

    save_npz(X, y, OUTPUT_NPZ)

    train_and_save_svm(X, y)
    print("All done.")


if __name__ == "__main__":
    main()
