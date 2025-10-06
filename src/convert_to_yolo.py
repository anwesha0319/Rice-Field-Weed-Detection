import os
import xml.etree.ElementTree as ET

# Define your classes in the same order as they appear in your XML annotations
# 'weed' will be class 0, and 'rice' will be class 1
classes = ['weed', 'rice']

def convert_coordinates(size, box):
    """
    Converts PASCAL VOC bounding box coordinates to YOLO format.
    size: (width, height) of the image
    box: (xmin, ymin, xmax, ymax) of the bounding box
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
    # Create the output directory if it doesn't exist
    os.makedirs(yolo_output_dir, exist_ok=True)

    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        image_size = root.find('size')
        
        # Check if size element exists
        if image_size is None:
            print(f"Skipping {xml_file_path}: 'size' element not found.")
            return

        image_width = int(image_size.find('width').text)
        image_height = int(image_size.find('height').text)

        output_txt_path = os.path.join(yolo_output_dir, os.path.basename(xml_file_path).replace('.xml', '.txt'))
        
        with open(output_txt_path, 'w') as out_file:
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                
                # Check if class name is valid
                if class_name not in classes:
                    print(f"Warning: Skipping unknown class '{class_name}' in {xml_file_path}")
                    continue
                
                class_id = classes.index(class_name)
                
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

if __name__ == '__main__':
    # Use the absolute path to your raw data's outputs folder
    RAW_XML_PATH = r"C:\Users\anwes\OneDrive\Desktop\22BCE10918_ANWESHA_RiceFieldWeedDetection\data\raw data\outputs"
    # This will create a 'labels' folder under 'raw data'
    YOLO_LABELS_PATH = r"C:\Users\anwes\OneDrive\Desktop\22BCE10918_ANWESHA_RiceFieldWeedDetection\data\raw data\labels"
    
    # Process all XML files in the raw data directory
    for file_name in os.listdir(RAW_XML_PATH):
        if file_name.endswith('.xml'):
            convert_xml_to_yolo(os.path.join(RAW_XML_PATH, file_name), YOLO_LABELS_PATH)
            
    print("XML to YOLO conversion complete.")
