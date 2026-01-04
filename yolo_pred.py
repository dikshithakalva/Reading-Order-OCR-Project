from ultralytics import YOLO
import json


class YOLOJsonExtractor:
    """
    A class to handle YOLO object detection and return bounding boxes in JSON format.
    """
    def __init__(self, model_path):
        """
        Initialize the YOLOJsonExtractor with a specified YOLO model.

        Args:
            model_path (str): Path to the YOLO model weights.
        """
        self.model = YOLO(model_path)

    def get_boxes_as_json(self, result, page_number):
        """
        Convert YOLO bounding boxes to JSON format with unique IDs.

        Args:
            result: YOLO model result object containing boxes.
            page_number (int): The page number corresponding to the detection.

        Returns:
            dict: JSON data with bounding boxes and unique IDs.
        """
        boxes = result.boxes  # YOLO result containing detected boxes
        xyxy_boxes = boxes.xyxy.cpu().numpy()  # Convert to numpy array (detach if on GPU)

        words = []
        for idx, box in enumerate(xyxy_boxes, start=1):
            x_min, y_min, x_max, y_max = box
            word_entry = {
                "id": idx,
                "bounding_box": {
                    "x_min": int(x_min),
                    "y_min": int(y_min),
                    "x_max": int(x_max),
                    "y_max": int(y_max)
                }
            }
            words.append(word_entry)

        output_data = {
            "page": page_number,
            "words": words
        }
        return output_data

    def process_image(self, image_path, page_number, conf=0.15, iou=0.15, img_size=(1024, 1024)):
        """
        Process an input image, run inference using YOLO, and return the results as JSON.

        Args:
            image_path (str): Path to the input image.
            page_number (int): Page number for the output JSON.
            conf (float): Confidence threshold for YOLO detection.
            iou (float): IoU threshold for YOLO detection.
            img_size (tuple): Image size to resize input images.

        Returns:
            dict: JSON data with bounding boxes and IDs for the image.
        """
        results = self.model(
            source=image_path,
            conf=conf,
            iou=iou,
            save_txt=False,
            save=False,
            imgsz=img_size,
            verbose=False
        )

        # Use the first result for processing (assuming single image input)
        result = results[0]
        return self.get_boxes_as_json(result, page_number)

    def save_json(self, output_data, output_path):
        """
        Save the extracted JSON data to a file.

        Args:
            output_data (dict): JSON data to be saved.
            output_path (str): Path to save the JSON file.
        """
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(output_data, json_file, indent=4, ensure_ascii=False)
        print(f"JSON output saved to: {output_path}")


if __name__ == "__main__":
    # Initialize the class with the model path
    model_path = "/home/dikshitha/ReadingOrder/best.pt"
    image_path = "/home/dikshitha/ReadingOrder/Image.jpg"
    output_json_path = "/home/dikshitha/ReadingOrder/output.json"
    page_number = 1

    # Create an instance of the class and process the image
    yolo_extractor = YOLOJsonExtractor(model_path)
    result_json = yolo_extractor.process_image(image_path, page_number)

    # Save the JSON output to a file
    yolo_extractor.save_json(result_json, output_json_path)
