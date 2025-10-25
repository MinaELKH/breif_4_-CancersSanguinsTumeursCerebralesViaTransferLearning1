Medical Image Analysis Project
==============================

Overview
--------

This project develops a unified deep learning solution for analyzing medical images to detect brain tumors from MRI or brain scans and classify abnormal blood cells for leukemia detection. It combines object detection using YOLOv8 for brain tumors and image classification using a modified GoogLeNet model in PyTorch for blood cells. The solution is integrated into an interactive Streamlit application for user-friendly analysis.

The project is divided into two main components:

*   **Blood Cell Classification**: Uses PyTorch to classify cancerous blood cells from blood smear images.
    
*   **Brain Tumor Detection**: Uses YOLOv8 to detect and localize brain tumors in brain scans.
    
*   **Streamlit Interface**: An application (app.py) that integrates both models for interactive use.
    

Features
--------

*   Data preprocessing: Handling image formats, splitting datasets into train/validation/test sets, applying transformations like blur, noise, and flip for data augmentation.
    
*   Model training and evaluation: Customizing pre-trained models, defining learning rates, loss functions, and optimizers; evaluating accuracy and generalization.
    
*   Data integrity checks: Ensuring images and labels match, filtering invalid data.
    
*   Hyperparameter configuration: YAML files for YOLOv8 with and without augmentations.
    
*   Model saving: Trained models are persisted for reuse.
    
*   Interactive UI: Streamlit app for uploading images and running inferences on both models.
    

Requirements
------------

*   Python 3.8+
    
*   PyTorch (for blood cell classification)
    
*   Ultralytics YOLOv8 (for brain tumor detection)
    
*   Streamlit (for the web interface)
    
*   Additional libraries: NumPy, Pandas, Matplotlib, Seaborn, OpenCV, Pillow
    

Install dependencies via:

text

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pip install torch torchvision torchaudio ultralytics streamlit numpy pandas matplotlib seaborn opencv-python pillow   `

Installation
------------

1.  git clone cd medical-image-analysis
    
2.  Install required packages (as listed above).
    
3.  Download datasets:
    
    *   Blood cell dataset: Place images in a directory structure where folder names represent classes.
        
    *   Brain tumor dataset: Include images and corresponding .txt label files for YOLO format.
        
4.  Prepare data:
    
    *   Run preprocessing scripts to split datasets and apply transformations.
        

Usage
-----

### Blood Cell Classification (PyTorch Part)

1.  Import necessary libraries and load the dataset, verifying image extensions (jpeg, jpg, bmp, png) and handling loading errors with try-except.
    
2.  Explore classes and display sample counts via countplot and image samples per class.
    
3.  Split data: 70% train, 15% validation, 15% test.
    
4.  Apply augmentations (blur, noise, flip) to balance classes.
    
5.  Use PyTorch transforms for resizing, tensor conversion, and normalization in ImageFolder datasets.
    
6.  Create DataLoaders for batching and shuffling.
    
7.  Load pre-trained GoogLeNet, modify the fully connected layer for classification.
    
8.  Set learning rate, loss function (e.g., CrossEntropyLoss), and optimizer (e.g., Adam).
    
9.  Train the model, evaluate on validation/test sets for accuracy and generalization.
    
10.  Save the trained model.
    

### Brain Tumor Detection (YOLOv8 Part)

1.  Display sample images with bounding boxes for each class.
    
2.  Create output directories for filtered data.
    
3.  Filter images: Copy images and matching .txt labels to train/valid/test folders; skip images without labels.
    
4.  Create data.yaml with paths, class count, and names (disable augmentations).
    
5.  Create data2.yaml with augmentations enabled.
    
6.  Count images and labels in sets; verify matches and remove mismatches.
    
7.  Train YOLOv8 with appropriate hyperparameters.
    
8.  Evaluate on validation/test sets for precision and generalization.
    
9.  Save the trained model.
    

### Streamlit Application

Run the Streamlit app to interact with both models:

text

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   streamlit run app.py   `

*   Upload blood smear images for classification.
    
*   Upload brain scans for tumor detection and localization.
    
*   View results, including predictions, confidence scores, and visualizations (e.g., bounding boxes for tumors).
    

Directory Structure
-------------------

*   data/: Raw datasets for blood cells and brain tumors.
    
*   output/: Processed data folders (train, valid, test for images and labels).
    
*   models/: Saved trained models.
    
*   scripts/: Preprocessing and training scripts.
    
*   app.py: Streamlit application for inference.