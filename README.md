# Aerial-Object-Classification-and-Detection
Problem statement: The project aims to develop a deep learning-based solution that can classify aerial images into two categories — Bird or Drone. The project involves building a Custom CNN classification model, leveraging transfer learning models like - resnet, mobilenet, effiecient net and the best model was deployed using streamlit for interactive use. 
Object-Classification:
Models used: Custom CNN, Resnet, Mobilenet, Effiecient net. 
Steps involved: 
              Data Collection
              Data structure Inspection
              Image counts per class
              Class Imbalance
              Visualize Sample images
              Image Preprocessing
                    Resize pixels to 224
                    Normalization
                    create dataset generators
              Verified preprocessed images by vewing sample images
              Defining Data Augmentation
              Model building - Custom CNN 
                    Model defining
                    Training
                    Evaluation
              Tranfer learning Models
                    Resnet model defining, training, Evaluation
                    Mobilenet model defining, training, Evaluation
                    Effiecient net model defining, training, Evaluation
              Model Predictions
              Metrics calculation
              Confusion matrix of best model
              best model fine tuning
              Best model saving
Bets model Chosen: Fine tuned Mobile net pretained model with 99% accuracy. 
Streamlit Deployment

Object Detection: 
            1. Model Used: Pretrained YOLOv8 (yolov8n.pt) for transfer learning
            2. Training
                  Configured using data.yaml
                  Trained with:
                  Epochs: 50 -- interuppted at 25
                  Image size 640
                  Output:
                  best.pt (best accuracy)
                  last.pt (latest checkpoint)
            3. Streamlit Deployment
              
                    
              
  
              
          

