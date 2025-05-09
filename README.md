Atom Vision
Atom Vision is a real-time surveillance system designed to enhance safety by detecting:

Face Masks: Ensures compliance with health guidelines.

Fire and Smoke: Provides early hazard detection.

Weapons: Identifies potential security threats.

Leveraging pre-trained ONNX models, Atom Vision ensures efficient and accurate detection suitable for various environments.

Table of Contents
Features

Model Details

Installation

Usage

Model Downloads

License

Features
Mask Detection: Utilizes MaskRCNN-12.onnx for instance segmentation to identify individuals wearing or not wearing face masks.

Fire/Smoke Detection: Employs tinyyolov2-8.onnx for rapid detection of fire and smoke instances.

Weapon Detection: Also uses tinyyolov2-8.onnx to identify potential weapons in the surveillance area.

Real-Time Processing: Optimized for real-time inference with minimal latency.

Modular Design: Easily extendable to incorporate additional detection capabilities.

Model Details
1. MaskRCNN-12.onnx
Purpose: Instance segmentation for detecting face masks.

Source: ONNX Model Zoo

Opset Version: 12

Input: Images of fixed size

Output: Segmentation masks, class labels, and bounding boxes

2. tinyyolov2-8.onnx
Purpose: Object detection for fire, smoke, and weapons.

Source: ONNX Model Zoo

Opset Version: 8

Input: Images of fixed size

Output: Bounding boxes and class probabilities

Installation
Clone the Repository

bash
Copy
Edit
git clone https://github.com/yourusername/atom-vision.git
cd atom-vision
Create a Virtual Environment (Optional but Recommended)

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
Note: Ensure that onnxruntime, opencv-python, and other dependencies are specified in requirements.txt.

Download the Pre-trained Models

Refer to the Model Downloads section below.

Usage
Run the Application

bash
Copy
Edit
python main.py
Provide Input

The application can process:

Live camera feed

Video files

Image files
MMDetection
Hugging Face
+1
PyPI
+1
Medium

Ensure that the input source is correctly specified in the configuration or command-line arguments.

View Output

The application will display the input with overlaid detections:

Bounding boxes for detected objects

Labels indicating the type of detection (e.g., "Mask", "Fire", "Weapon")

Confidence scores
Medium
+10
PyPI
+10
GitHub
+10

Model Downloads
The pre-trained models used by Atom Vision are available from the ONNX Model Zoo.

1. MaskRCNN-12.onnx
Download Link: MaskRCNN-12.onnx

Instructions:

bash
Copy
Edit
wget https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-12.onnx
Place the downloaded file in the models/ directory of the project.

2. tinyyolov2-8.onnx
Download Link: tinyyolov2-8.onnx

Instructions:

bash
Copy
Edit
wget https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.onnx
Place the downloaded file in the models/ directory of the project.

Note: If you encounter issues with wget due to GitHub's file hosting, consider downloading the files manually through your web browser.

License
This project is licensed under the MIT License.

