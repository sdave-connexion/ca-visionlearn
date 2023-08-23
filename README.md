
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

