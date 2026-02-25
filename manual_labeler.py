import os
import cv2
import shutil
import csv

DATASET_MODE1 = "dataset_1"
DATASET_MODE2 = "dataset"
OUTPUT_FILE = "labels.csv"
MAX_WIDTH = 1000
MAX_HEIGHT = 800


def resize_to_fit(img, max_width, max_height):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    return cv2.resize(img, (int(w*scale), int(h*scale)))

def get_letter_from_user():
    while True:
        key = cv2.waitKey(0)
        if key == 27:  # ESC
            return None
        if 65 <= key <= 90 or 97 <= key <= 122:
            return chr(key).upper()
        print(" Invalid key. Press a letter A–Z.")

def confirm_label():
    while True:
        print("Confirm? (Y/N)")
        key = cv2.waitKey(0)
        if key in [ord('y'), ord('Y')]:
            return True
        elif key in [ord('n'), ord('N')]:
            return False
        else:
            print(" Press Y for yes or N for no.")

def extract_frames_to_dataset1():
    video_path = input("Enter path to video file: ").strip()

    if not os.path.exists(video_path):
        print("Video file not found!")
        return

    os.makedirs(DATASET_MODE1, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video!")
        return

    frame_count = 0
    saved_count = 0

    print("Extracting frames…")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 5 != 0:
            continue
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Save every frame as a JPG
        filename = f"frame_{frame_count:06d}.jpg"
        save_path = os.path.join(DATASET_MODE1, filename)

        cv2.imwrite(save_path, frame)
        saved_count += 1

    cap.release()
    print(f"Done! Extracted {saved_count} frames into '{DATASET_MODE1}'")

def mode1_build_dataset():
    if not os.path.exists(DATASET_MODE1):
        print(f"Folder '{DATASET_MODE1}' not found!")
        return

    images = [os.path.join(DATASET_MODE1, f)
              for f in os.listdir(DATASET_MODE1)
              if f.lower().endswith(("png","jpg","jpeg"))]

    if not images:
        print(f"No images found in {DATASET_MODE1}")
        return

    print(f"Found {len(images)} images in unorganized folder.\n")

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Cannot open {img_path}")
            continue

        img_display = resize_to_fit(img, MAX_WIDTH, MAX_HEIGHT)
        window_name = "Assign Letter (Mode 1)"
        cv2.imshow(window_name, img_display)
        cv2.waitKey(1)

        while True:
            letter = get_letter_from_user()
            if letter is None:
                cv2.destroyAllWindows()
                print("Exiting...")
                return

            print(f"You typed: {letter}")
            if confirm_label():
                # Create folder if needed
                folder_path = os.path.join(DATASET_MODE2, letter)
                os.makedirs(folder_path, exist_ok=True)
                # Move image
                dst_path = os.path.join(folder_path, os.path.basename(img_path))
                shutil.copy2(img_path, dst_path)
                print(f"Saved {img_path} -> {dst_path}")
                break
            else:
                print("Type the correct letter.")

    cv2.destroyAllWindows()
    print("Dataset creation complete!")

def mode2_learning_app():

    if not os.path.exists(DATASET_MODE2):
        print(f"Folder '{DATASET_MODE2}' not found!")
        return

    label_folders = [d for d in os.listdir(DATASET_MODE2)
                     if os.path.isdir(os.path.join(DATASET_MODE2, d))]

    if not label_folders:
        print("No subfolders found in dataset.")
        return

    label_folders = sorted([f.upper() for f in label_folders])
    print("Found folders:", label_folders)

    images = []
    for label in label_folders:
        folder = os.path.join(DATASET_MODE2, label)
        for f in os.listdir(folder):
            if f.lower().endswith(("png","jpg","jpeg")):
                images.append((os.path.join(folder,f), label))

    if not images:
        print("No images found inside dataset folders.")
        return

    print(f"Loaded {len(images)} images.\n")
    correct_count = 0
    total_count = 0

    for img_path, correct_label in images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Cannot open {img_path}")
            continue

        img_display = resize_to_fit(img, MAX_WIDTH, MAX_HEIGHT)
        window_name = "Learning Mode (Mode 2)"
        cv2.imshow(window_name, img_display)
        cv2.waitKey(1)

        print(f"\nImage: {img_path}")
        print(f"Correct label (for internal reference): {correct_label}")

        while True:
            typed_letter = get_letter_from_user()
            if typed_letter is None:  # ESC
                cv2.destroyAllWindows()
                print("\nExiting learning mode…")
                print(f"Your score: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)")
                return

            total_count += 1
            if typed_letter == correct_label.upper():
                correct_count += 1
                print(f" Correct! ({typed_letter})")
            else:
                print(f" Incorrect! You typed '{typed_letter}', correct is '{correct_label}'")

            # Move to next image
            break

    cv2.destroyAllWindows()
    print(f"\nLearning session finished!")
    print(f"Your score: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)")

def main():
    print("Choose mode:")
    print("0 - Extract frames from video into dataset_1")
    print("1 - Build dataset from unorganized images (dataset_1)")
    print("2 - Label already organized dataset (dataset)")
    choice = input("Enter 0, 1 or 2: ")

    if choice == "0":
        extract_frames_to_dataset1()
    elif choice == "1":
        mode1_build_dataset()
    elif choice == "2":
        mode2_learning_app()
    else:
        print("Invalid choice. Exiting.")




if __name__ == "__main__":
    main()
