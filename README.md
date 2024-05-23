# Image-Segmentation-on-Custom-dataset
The project focuses on developing a model capable of detecting vehicle.<br/>
<br/>
![1 (1)](https://github.com/meerap1/Image-Segmentation-on-Custom-dataset/assets/156745402/e0cdbd05-d1c2-49da-96a1-805fed63845c)

## Table of Content
1. Introduction
2. Data Source
3. Data Segmentation
4. Creating Virtual Environment
5. Setup Yolov8 Repository
6. Train Yolov8 Model
7. Testing on Images and Videos
8. Results
9. Inferences

### Introduction
I'm excited to share with you my latest project: a YOLOv8 model for custom data. Leveraging state-of-the-art deep learning techniques, this model is designed to accurately identify and localize type of vehicle in images. Utilizing Google colab, I've crafted an environment where you can easily set up and run the model on your own machine.

### Data Source:
The data used in this repository is sourced from Kaggle and other open sources. A subset of over 477 images and labels from this dataset was selected for training purposes. This curated dataset includes images, where positive samples feature various types of vehicles on roads, annotated with bounding boxes and class labels.

### Data Segmentation:
Data segmentation is performed using Roboflow. The dataset includes 6 classes: 'Auto', 'Bus', 'Car', 'Traveller', 'Truck', and 'Two Wheeler'. The data is divided into training and validation sets in a 90:10 ratio.

### Creating Virtual Environement:
First, change the runtime to **T4 GPU** and connect it. Then, check  to ensure that the GPU is properly configured and available for use. <br/>
<br/>
**`!nvidia-smi`** <br/>
<br/>
The following code is used to mount Google Drive in a Google Colab environment: <br/>
<br/>
**`from google.colab import drive`**<br/>
<br/>
**`drive.mount('/content/gdrive')`** <br/>
<br/>
The following commands are used to navigate directories in Google Colab environment:<br/>
<br/>
**`%cd yolov8`**<br/>
<br/>
**`%cd custom_dataset`** <br/>
<br/>
In the custom_dataset folder, we have training and validation datasets along with a data.yaml file. The data.yaml file typically contains information about the dataset, such as the number of classes, class names, and paths to the training and validation data. <br/>
<br/>
![Screenshot 2024-05-24 013518](https://github.com/meerap1/Image-Segmentation-on-Custom-dataset/assets/156745402/80cc754a-8617-41f9-91c1-1bba26e1f213)

By navigating to this directory, we are setting up the environment to access and use these files for training the YOLOv8 model.
### Setup Yolov8 Repository
The following command installs the ultralytics package:<br/>
<br/>
**`!pip install ultralytics`** <br/>
### Train Yolov8 Model
The following command is used to train a YOLOv8 segmentation model:<br/>
<br/>
**`!yolo task=segment mode=train epochs=100 data=data.yaml model=yolov8m-seg.pt imgsz=640 batch=8`** <br/>
<br/>
This command initiates the training of a YOLOv8 segmentation model using the specified pre-trained model (yolov8m-seg.pt) on THE custom dataset defined in data.yaml, with the images resized to 640x640 pixels, and a batch size of 8 for 100 epochs. <br/>
<br/>
After the training process, the best-performing model weights are saved as best.pt in the directory runs/segment/train/weights/. This best.pt file represents the model with the highest validation performance during training. To utilize this trained model for future inference or further training, it is common practice to copy best.pt and paste it into the custom_data folder, renaming it to yolov8m-seg-custom.pt. This ensures that the most optimized version of the model is readily accessible and identified within the custom dataset directory.

### Testing on Images and Videos
To prepare for making predictions using a trained YOLOv8 segmentation model, create this py file in custom_dataset and name it predict.py.:<br/>
<br/>
![Screenshot 2024-05-24 015023](https://github.com/meerap1/Image-Segmentation-on-Custom-dataset/assets/156745402/da12e682-d816-4f9c-bd15-93f053285cc0) <br/>
<br/>
This script initializes a YOLOv8 model using the specified weights file (yolov8m-seg-custom.pt). It then makes predictions on the image 2.jpg. Parameters like show, save, hide_conf, conf, save_txt, save_crop, and line_thickness control various aspects of the prediction process, such as displaying the prediction, saving the prediction, hiding confidence scores, setting confidence threshold, saving prediction results in text format, saving cropped objects, and adjusting line thickness in the visualization, respectively. Adjust these parameters based on your specific requirements.<br/>
<br/>
The output will be saved in runs/segment/predict/.

### Result
The output is generated in both image and video formats.<br/>

![1 (1)](https://github.com/meerap1/Image-Segmentation-on-Custom-dataset/assets/156745402/58764908-6f60-4a23-9758-701ad3a49c6b) </br>
 </br>
 ![2 (1)](https://github.com/meerap1/Image-Segmentation-on-Custom-dataset/assets/156745402/e23552c2-6df7-4224-8537-a488defb6a4c)
 
### Inferences
In evaluating inferences, which employed two key metrics: the confusion matrix and the F1 curve. The confusion matrix provides a comprehensive overview of the model's performance across different classes, illustrating its ability to correctly classify instances. Notably, while some classes showed comparetively better precision scores, such as 'Bus' and 'Car', others, like 'Truck' and 'Two Wheeler', had less conclusive results. <br/>
<br/>
![confusion_matrix_normalized](https://github.com/meerap1/Image-Segmentation-on-Custom-dataset/assets/156745402/76c2e860-4329-4d02-a327-d4e02feaded1) <br/>
<br/>
The F1 curve, on the other hand, presents a holistic view of the model's precision and recall trade-offs across various confidence thresholds, offering insights into its overall performance trends. <br/>
<br/>
![BoxF1_curve](https://github.com/meerap1/Image-Segmentation-on-Custom-dataset/assets/156745402/2cb018ae-8086-424d-b2e5-00e7a42870f1) <br/>
<br/>
In conclusion, the model demonstrates competence in certain areas, there are clear opportunities for enhancement, particularly in refining its ability to accurately identify specific classes.
















