from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import requests
import tensorflow as tf
import numpy as np
import os
import cv2
import numpy as np
from skimage.segmentation import felzenszwalb
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
import cv2
import numpy as np
import joblib
from skimage.segmentation import felzenszwalb
from skimage.color import rgb2gray
from skimage.measure import regionprops
from skimage.feature import local_binary_pattern
import sklearn
import requests
from flask import jsonify
from PIL import Image, UnidentifiedImageError
import cv2
import numpy as np
import io
from datetime import datetime
import requests
from flask import jsonify
from PIL import Image, UnidentifiedImageError  # Explicitly import UnidentifiedImageError
import cv2
import numpy as np
import io
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for the app

errorMessage = None
model = None

# Load your pre-trained model with error handling
try:    
    model = tf.keras.models.load_model('models/Rebuilt_Xception')
    print("Model loaded successfully!")
except Exception as e:    
    errorMessage = e
    model = None
    
try:
    model_path = 'models/ClassifierModel/Classifer_Using_Thresholding_and_Decision_Tree_2.plk'
    classifier_model = joblib.load(model_path)
    print("Classifier model loaded successfully!")
except Exception as e:
    errorMessage = e
    model = None

# Function to preprocess the image
def preprocess_image(image, target_size=(128, 128)):
    # Convert image to RGB format
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    # Convert to PIL Image format for resizing
    img_pil = Image.fromarray(img_rgb)
    # Resize the image
    img_resized = img_pil.resize(target_size)
    # Convert to NumPy array and normalize
    img_array = np.array(img_resized) / 255.0  
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)  
    
    return img_array

sub_image_shape = []

def strideImage(image):
    startx, starty = 300, 350
    endx, endy = 1100, 1150
    cropped_image = image[starty:endy, startx:endx]
    
    image = cropped_image
    height, width, _ = image.shape

    grid_size = 6

    box_width = width // grid_size
    box_height = height // grid_size

    grid_boxes = []

    for i in range(grid_size):
        row = []
        for j in range(grid_size):
            x_start = j * box_width
            y_start = i * box_height
            x_end = x_start + box_width
            y_end = y_start + box_height
            row.append(image[y_start:y_end, x_start:x_end])
        grid_boxes.append(row)
        
    def sliding_window(grid_boxes, grid_size, window_size):
        stride = 1
        boxes = []
        for i in range(0, grid_size - window_size + 1, stride):
            for j in range(0, grid_size - window_size + 1, stride):
                sub_grid = [grid_boxes[i+k][j:j+window_size] for k in range(window_size)]
                sub_image = np.vstack([np.hstack(row) for row in sub_grid])
                sub_height, sub_width, _ = sub_image.shape
                global sub_image_shape
                sub_image_shape = [sub_height , sub_width]
                boxes.append(sub_image)

        return boxes

    boxes_2x2 = sliding_window(grid_boxes, grid_size=6, window_size=2)
    boxes_3x3 = sliding_window(grid_boxes, grid_size=6, window_size=2)
    return boxes_2x2, boxes_3x3

def load_image(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    return cv2.resize(image, (64, 64)) if image is not None else None

def calculate_white_pixel_ratio(gray_image):
    _, thresholded_image = cv2.threshold((gray_image * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    white_pixel_count = np.sum(thresholded_image == 255)
    return white_pixel_count / thresholded_image.size

def extract_node_features(image):
    segments = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
    node_features = []
    for region_id in np.unique(segments):
        mask = segments == region_id
        if mask.sum() == 0:
            continue

        props = regionprops(mask.astype(int), intensity_image=rgb2gray(image))
        if props:
            node_features.append(extract_region_features(image, mask, props[0]))
    return node_features

def extract_region_features(image, mask, region_props):
    mean_color = np.mean(image[mask], axis=0)
    area, perimeter = region_props.area, region_props.perimeter
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    colorfulness = np.sqrt((mean_color[0] - mean_color[1]) ** 2 + (0.5 * (mean_color[0] + mean_color[1]) - mean_color[2]) ** 2)

    # LBP histogram
    lbp = local_binary_pattern(rgb2gray(image), P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp[mask], bins=np.arange(0, 11), density=True)

    centroid = region_props.centroid
    return [centroid[0] / image.shape[0], centroid[1] / image.shape[1], *mean_color, area, perimeter, circularity, colorfulness, *lbp_hist]

def process_image(image):
    gray_image = rgb2gray(image)
    white_pixel_ratio = calculate_white_pixel_ratio(gray_image)
    node_features = extract_node_features(image)


    if node_features:
        return np.mean(node_features, axis=0).tolist() + [white_pixel_ratio]
    return None

# Function to predict the cyclone intensity
def predict_image(img):
    if model is None:
        return None, f"Model is not loaded. Error: {str(errorMessage)}"

    # Preprocess and predict the image
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    predicted_value = prediction[0][0]  # Since it's regression, extract the first value
    
    return predicted_value, None

def predict_class(img_path, classifier):
    image = load_image(img_path)

    if image is None:
        print("Image not found.")
        return "", None

    feature_vector = process_image(image)
    if feature_vector is None:
        print("No features extracted.")
        return "", None


    prediction = classifier.predict([feature_vector])[0]
    probability = classifier.predict_proba([feature_vector])[0][prediction]
    class_label = "Cyclone" if prediction == 0 else "Non-Cyclone"
    print(f"Prediction: {class_label}, Probability: {probability:.2f}")

    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.title(f"Prediction: {class_label}, Probability: {probability:.2f}")
    # plt.show()

    return class_label, probability

# Route for the homepage
@app.route('/intensity-estimation')
def home():
    return render_template('index.html')

@app.route('/cyclone-detection-and-estimation')
def home2():
    return render_template('index2.html')

@app.route('/')
def home3():
    return render_template('index3.html')

@app.route('/predictWithClassify', methods=['POST'])
def predictWithClassify():
    if model is None:
        return jsonify({"error": str(errorMessage)})

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file and file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
        big_one = Image.open(file.stream)
        img = cv2.cvtColor(np.array(big_one), cv2.COLOR_RGB2BGR)

        # Apply cropping
        startx, starty = 300, 350
        endx, endy = 1100, 1150
        cropped_image = img[starty:endy, startx:endx]

        # Classification logic
        flag = 0
        strides1, _ = strideImage(img)  # Process the cropped image
        predicted_output = None
        coordinates = []  # List to store the coordinates of cyclone regions

        grid_size = 6  # The number of divisions in the grid (6x6)
        box_width = cropped_image.shape[1] // grid_size
        box_height = cropped_image.shape[0] // grid_size

        # Sliding window logic - calculate the grid-based windows
        for i, box in enumerate(strides1):
            tempImagePath = "tempImages/tempImg.jpg"
            savepath = "static/cyclone.jpg"
            cv2.imwrite(tempImagePath, box)
            curClass, _ = predict_class(tempImagePath, classifier_model)
            cv2.imwrite(savepath,img)
            if curClass == 'Cyclone':
                flag = 1
                predicted_output, error = predict_image(load_image(tempImagePath))
                if error:
                    return jsonify({'error': error})

                row = i // grid_size 
                col = i % grid_size  
                
                x_start = (col * box_width) + startx
                y_start = (row * box_height) + starty

                x_end = x_start + sub_image_shape[1]
                y_end = y_start + sub_image_shape[0]

                coordinates.append({
                    'x_start': x_start,
                    'y_start': y_start,
                    'x_end': x_end,
                    'y_end': y_end
                })
                cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (0,0,255), 2)
                label_position = (x_end, y_start - 5) 
                cv2.putText(img, str(predicted_output), label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                cv2.imwrite(savepath,img)
                break  

        if not flag:
            return jsonify({
                'classification': 'No cyclone detected',
                'intensity': None,
                'coordinates': coordinates,
                'indexes': "Try uploading another image."
            })

        return jsonify({
            'classification': 'Cyclone detected',
            'intensity': float(predicted_output),
            'coordinates': coordinates,
            'indexes': len(strides1)
        })
    
    return jsonify({'error': 'Invalid file format. Please upload an image file (png, jpg, jpeg).'})




@app.route('/predictWithoutClassify', methods=['POST'])
def predictWithoutClassify():
    print("Received request for predictWithoutClassify")  # Log request received
    
    if model is None:
        print("Model is not loaded")  # Log missing model
        return jsonify({"error": str(errorMessage)})

    if 'file' not in request.files:
        print("No file in request")  # Log missing file
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    print(f"File received: {file.filename}")  # Log file name
    
    if file and file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
        try:
            big_one = Image.open(file.stream)
            img = cv2.cvtColor(np.array(big_one), cv2.COLOR_RGB2BGR)
            
            print("Image converted to BGR")  # Log image conversion
            
            predicted_output, error = predict_image(img)

            if error:
                print(f"Prediction error: {error}")  # Log prediction error
                return jsonify({'error': error})

            print(f"Predicted output: {predicted_output}")  # Log successful prediction
            return jsonify({'intensity': float(predicted_output)})

        except Exception as e:
            print(f"Exception occurred: {e}")  # Log any unexpected exceptions
            return jsonify({'error': 'Internal server error. Please try again.'})

    print("Invalid file format")  # Log invalid file type
    return jsonify({'error': 'Invalid file format. Please upload an image file (png, jpg, jpeg).'})

IMAGE_URL = "https://mausam.imd.gov.in/Satellite/3Dasiasec_ir1.jpg"

@app.route('/fetchAndPredict', methods=['GET'])
def fetch_and_predict():
    savepath = "static/cyclone.jpg"
    if model is None:
        return jsonify({"error": str(errorMessage)})

    try:
        # Fetch the image from the URL
        response = requests.get(IMAGE_URL, stream=True)
        response.raise_for_status()  # Check for HTTP errors

        # Ensure the response is an image by attempting to open it
        try:
            image = Image.open(io.BytesIO(response.content))
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cv2.imwrite(savepath,img)
            
        except UnidentifiedImageError:
            return jsonify({"error": "The fetched content is not a valid image."})

        startx, starty = 300, 350
        endx, endy = 1100, 1150
        cropped_image = img[starty:endy, startx:endx]

        coordinates = []  

        grid_size = 6  
        box_width = cropped_image.shape[1] // grid_size
        box_height = cropped_image.shape[0] // grid_size

        # Process image and predict
        flag = 0
        strides1, _ = strideImage(img)
        for i,box in enumerate(strides1):
            temp_image_path = "tempImages/tempImg.jpg"
            cv2.imwrite(temp_image_path, box)
            cur_class, _ = predict_class(temp_image_path, classifier_model)
            if cur_class == 'Cyclone' and _ >= 0.85:
                flag = 1
                predicted_output, error = predict_image(load_image(temp_image_path))
                
                row = i // grid_size 
                col = i % grid_size  
                
                x_start = (col * box_width) + startx
                y_start = (row * box_height) + starty

                x_end = x_start + sub_image_shape[1]
                y_end = y_start + sub_image_shape[0]

                coordinates.append({
                    'x_start': x_start,
                    'y_start': y_start,
                    'x_end': x_end,
                    'y_end': y_end
                })
                cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (0,0,255), 2)
                label_position = (x_end, y_start - 5) 
                cv2.putText(img, str(predicted_output), label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                cv2.imwrite(savepath,img)

                break
        if not flag:
            return jsonify({'intensity': 'No cyclone detected in the image', 'timestamp': str(datetime.now())})

        if error:
            return jsonify({'error': error})

        return jsonify({'intensity': float(predicted_output), 'timestamp': str(datetime.now())})

    except requests.RequestException as e:
        return jsonify({'error': f'Failed to fetch image: {str(e)}'})
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'})
    


# if __name__ == '__main__':
#     app.run(host="localhost",port=3000,debug=True)