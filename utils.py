import numpy as np
from tensorflow.keras.preprocessing import image

def get_disposal_suggestion(prediction_class):
    suggestions = {
        'Biodegradable': {
            'action': 'Compost',
            'details': 'This item can be decomposed naturally. Add it to a compost bin to create nutrient-rich soil.',
            'color': '#28a745'
        },
        'Non-Biodegradable': {
            'action': 'Proper Waste Disposal',
            'details': 'This item does not decompose easily. Dispose of it in the general waste bin according to local regulations.',
            'color': '#dc3545'
        },
        'Recyclable': {
            'action': 'Recycle Bin',
            'details': 'This item can be processed and reused. Clean it if necessary and place it in the appropriate recycling bin.',
            'color': '#007bff'
        }
    }
    return suggestions.get(prediction_class, {'action': 'Unknown', 'details': 'No suggestion available.', 'color': '#6c757d'})

def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array
