import os
import shutil
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from pathlib import Path

# --- Configuration ---
# Use the full, absolute paths you provided
RAW_DATA_PATH = r"C:\Users\anwes\OneDrive\Desktop\22BCE10918_ANWESHA_RiceFieldWeedDetection\data\raw data"
PROCESSED_DATA_PATH = r"C:\Users\anwes\OneDrive\Desktop\22BCE10918_ANWESHA_RiceFieldWeedDetection\data\processed_data"

# Define your classes in the same order as they appear in your XML annotations
# 'weed' will be class 0, and 'rice' will be class 1
CLASSES = ['weed', 'rice']

# --- XML to YOLO Conversion Functions ---
def convert_coordinates(size, box):
    """
    Converts PASCAL VOC bounding box coordinates to YOLO format.
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x_center = x_center * dw
    w = w * dw
    y_center = y_center * dh
    h = h * dh
    return (x_center, y_center, w, h)

def convert_xml_to_yolo(xml_file_path, yolo_output_dir):
    """
    Reads an XML file and creates a corresponding YOLO .txt file.
    """
    os.makedirs(yolo_output_dir, exist_ok=True)
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        image_size = root.find('size')
        if image_size is None:
            print(f"Skipping {xml_file_path}: 'size' element not found.")
            return

        image_width = int(image_size.find('width').text)
        image_height = int(image_size.find('height').text)
        
        output_txt_path = os.path.join(yolo_output_dir, Path(xml_file_path).stem + '.txt')
        
        with open(output_txt_path, 'w') as out_file:
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name not in CLASSES:
                    print(f"Warning: Skipping unknown class '{class_name}' in {xml_file_path}")
                    continue
                
                class_id = CLASSES.index(class_name)
                
                box = obj.find('bndbox')
                xmin = float(box.find('xmin').text)
                ymin = float(box.find('ymin').text)
                xmax = float(box.find('xmax').text)
                ymax = float(box.find('ymax').text)

                b_coords = (xmin, xmax, ymin, ymax)
                yolo_coords = convert_coordinates((image_width, image_height), b_coords)
                
                out_file.write(f"{class_id} {' '.join([str(a) for a in yolo_coords])}\n")

    except Exception as e:
        print(f"Error processing {xml_file_path}: {e}")

# --- Data Splitting and Organization Functions ---
def split_and_organize_data(raw_images_dir, raw_labels_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Splits images and labels and organizes them into the correct YOLOv8 format.
    """
    if not os.path.exists(raw_images_dir):
        raise FileNotFoundError(f"Raw images directory not found at: {raw_images_dir}")
    if not os.path.exists(raw_labels_dir):
        raise FileNotFoundError(f"Raw labels directory not found at: {raw_labels_dir}")

    all_images = [f for f in os.listdir(raw_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not all_images:
        raise ValueError(f"No image files found in the directory: {raw_images_dir}")

    train_images, remaining_images = train_test_split(all_images, test_size=(val_ratio + test_ratio), random_state=42)
    val_images, test_images = train_test_split(remaining_images, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)

    for folder in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', folder), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', folder), exist_ok=True)
    
    print("Moving files to their respective folders...")
    
    def move_files(file_list, type_folder):
        for image_name in file_list:
            label_name = Path(image_name).stem + '.txt'
            shutil.copy(os.path.join(raw_images_dir, image_name), os.path.join(output_dir, 'images', type_folder, image_name))
            shutil.copy(os.path.join(raw_labels_dir, label_name), os.path.join(output_dir, 'labels', type_folder, label_name))

    move_files(train_images, 'train')
    move_files(val_images, 'val')
    move_files(test_images, 'test')
    print(f"Dataset split complete. Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")


if __name__ == '__main__':
    # --- Step 1: Convert XML to YOLO format ---
    print("Starting XML to YOLO conversion...")
    RAW_XML_PATH = os.path.join(RAW_DATA_PATH, 'outputs')
    YOLO_LABELS_PATH = os.path.join(RAW_DATA_PATH, 'labels')
    
    if not os.path.exists(RAW_XML_PATH):
        raise FileNotFoundError(f"XML annotations folder not found at: {RAW_XML_PATH}")

    for file_name in os.listdir(RAW_XML_PATH):
        if file_name.endswith('.xml'):
            convert_xml_to_yolo(os.path.join(RAW_XML_PATH, file_name), YOLO_LABELS_PATH)
            
    print("XML to YOLO conversion complete. Labels are in the 'labels' folder under raw data.")
    print("-" * 50)
    
    # --- Step 2: Split and Organize Data ---
    print("Starting data splitting and organization...")
    RAW_IMAGES_PATH = os.path.join(RAW_DATA_PATH, 'images')
    RAW_LABELS_PATH_FINAL = YOLO_LABELS_PATH
    
    split_and_organize_data(RAW_IMAGES_PATH, RAW_LABELS_PATH_FINAL, PROCESSED_DATA_PATH)
    print("-" * 50)
    print("All data processing is complete. You can now start training your YOLOv8 model.")