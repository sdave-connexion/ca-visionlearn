
# ca-visionlearn

The `ca-visionlearn` package offers a suite of tools tailored for image analysis, including color extraction, metric computation, skin tone identification, and more.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Image Color Analyzer](#image-color-analyzer)
3. [ImageMetrics](#imagemetrics)
4. [ProductColor](#productcolor)
5. [skin_tone Extraction](#skin_tone-extraction)
6. [Dependencies](#dependencies)

---

### Getting Started

Welcome to the `ca-visionlearn` starter guide. This notebook will walk you through the basic usage of the `ca-visionlearn` package using a sample image.

---

### Installation
```python
!pip install ca-visionlearn==3.1
```

---

### Importing Modules
```python
from ca_visionlearn.color_utils import ImageColorAnalyzer
from ca_visionlearn.metrics import ImageMetrics
from ca_visionlearn.product_color_processor import ProductColor
from ca_visionlearn.skin_tone_extraction import skin_tone
from PIL import Image
import matplotlib.pyplot as plt
import cv2
```

---

### Importing Dataset
```python
# Load the image
image_path = '/content/blue_dress.JPG'
image_array = cv2.imread(image_path) #OpenCV loads images in BGR by default and please do not change it
```

---

### Image Color Analyzer
...

(remaining content unchanged from provided text)

get_color_information`: Fetches color distribution data from a clustered image.
- `extract_dominant_color`: Draws out the dominant color from an image.
- `plot_color_bar`: Produces a color bar grounded in color distribution.
- `pretty_print_data`: Displays color data in a user-friendly manner.
- `display_results`: Shows the original, thresholded, and color bar images.
- `get_rgb`: Gives the RGB value of the dominant skin tone.

**Usage**:
Begin by importing the `skin_tone` class. Once done, you can create an instance by providing the image path. After that, the dominant RGB value of the skin tone can be obtained:

```python
# Create an instance of the skin_detector class
skin_detector = skin_tone(image_array)

# Visualize the results including original image, thresholded image, and color bar
skin_detector.display_results()

# Get the RGB value of the skin tone
dominant_skin = skin_detector.get_dominant_skin_tone()
print(dominant_skin)

```

---

### Dependencies

Ensure the following Python libraries are installed in your environment:
- numpy
- cv2 (OpenCV)
- PIL (Pillow)
- skimage

