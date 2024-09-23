from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg') 
from matplotlib import pyplot as plt
from io import BytesIO
import base64
from keras.models import load_model

app = Flask(__name__)

def watershed_segmentation(image_content):
   
    nparr = np.frombuffer(image_content.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    median_filtered_image = cv2.medianBlur(binary_image, ksize=5)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened_image = cv2.morphologyEx(median_filtered_image, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opened_image, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opened_image, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown_region = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)

    markers = markers + 1

    markers[unknown_region == 255] = 0

    segmented_cells = cv2.watershed(image, markers)
    image[segmented_cells == -1] = [255, 0, 0]

    num_cells_detected = np.max(segmented_cells) - 1

    fig, ax = plt.subplots(figsize=(7.5, 6))

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
   # plt.title('Original Image')
    plt.axis('off')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    

    # Convert the plot to a base64-encoded image
    buffered = BytesIO()
    plt.savefig(buffered, format="png")
    plt.close()

    original_image_str = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

    fig, ax = plt.subplots(figsize=(7.5, 6))

    plt.imshow(segmented_cells, cmap='jet')
   # plt.title('Segmented Image')
    plt.axis('off')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    buffered_segmented = BytesIO()
    plt.savefig(buffered_segmented, format="png")
    plt.close()

    segmented_image_str = "data:image/png;base64," + base64.b64encode(buffered_segmented.getvalue()).decode("utf-8")

    return original_image_str, segmented_image_str, num_cells_detected

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['image']

        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file:
            try:
                original_image_str, segmented_image_str, num_cells_detected = watershed_segmentation(file)
                return render_template('index.html', original_image=original_image_str, segmented_image=segmented_image_str, num_cells=num_cells_detected)
            except Exception as e:
                return render_template('index.html', error=f'Error processing the image: {str(e)}')

    return render_template('index.html', original_image=None, segmented_image=None, num_cells=None, error=None)

# Load the VGG16 model
model = load_model('C:/Users/nabas/WBC_segment/mobileNetV2_seg_data_model.h5')

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    image = np.expand_dims(image, axis=0)
    return image

def classify_cells(image, model, min_cell_area=500):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through each contour
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_cell_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cell = image[y:y+h, x:x+w]
        preprocessed_cell = preprocess_image(cell)

        # Perform prediction using the model
        predictions = model.predict(preprocessed_cell)
        cell_class = "Normal Cell" if predictions[0][0] > 0.5 else "Cancer Cell"

        # bounding box cancer image
        if cell_class == "Cancer Cell":
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        font_scale = 1.0
        font_thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, cell_class, (x, y - 10), font, font_scale, color, font_thickness, cv2.LINE_AA)

    return image

@app.route('/classification', methods=['GET', 'POST'])
def classification():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('classification.html', error='No file part')
        
        file = request.files['image']
        
        if file.filename == '':
            return render_template('classification.html', error='No selected file')

        try:
            nparr = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Apply watershed segmentation
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            median_filtered_image = cv2.medianBlur(binary_image, ksize=5)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            opened_image = cv2.morphologyEx(median_filtered_image, cv2.MORPH_OPEN, kernel, iterations=2)
            sure_bg = cv2.dilate(opened_image, kernel, iterations=3)
            dist_transform = cv2.distanceTransform(opened_image, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            unknown_region = cv2.subtract(sure_bg, sure_fg)
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown_region == 255] = 0
            segmented_cells = cv2.watershed(image, markers)
            segmented_mask = np.zeros_like(segmented_cells, dtype=np.uint8)
            segmented_mask[segmented_cells > 1] = 255
            transparent_image = np.zeros_like(image, dtype=np.uint8)
            transparent_image[:, :] = (255, 255, 255)  
            transparent_image[segmented_mask > 0] = image[segmented_mask > 0]

            # Classify cells in the image
            classified_image = classify_cells(transparent_image, model)

            classified_image_rgb = cv2.cvtColor(classified_image, cv2.COLOR_BGR2RGB)

            # Convert the classified image to base64 format for embedding in HTML
            _, buffer = cv2.imencode('.jpg', classified_image_rgb)
            classified_image_str = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

            return render_template('classification.html', classified_image=classified_image_str)
        except Exception as e:
            return render_template('classification.html', error=f'Error processing the image: {str(e)}')

    return render_template('classification.html')

@app.route('/index')
def segmentation():
    return redirect(url_for('index'))

@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
