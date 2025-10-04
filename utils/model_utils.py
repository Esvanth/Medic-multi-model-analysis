import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer
import cv2

class MedicalMultiModalPredictor:
    def __init__(self, model_path=None):
        self.model = None
        self.tokenizer = None
        self.image_size = 224
        self.diseases = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
            'Pleural Thickening', 'Hernia'
        ]
        
        self.load_tokenizer()
        if model_path:
            self.load_model(model_path)
    
    def load_tokenizer(self):
        """Load Bio_ClinicalBERT tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                'emilyalsentzer/Bio_ClinicalBERT'
            )
            print("Tokenizer loaded successfully")
        except:
            print("Using fallback tokenizer")
            from transformers import BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def load_model(self, model_path):
        """Load the trained multi-modal model"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = self.create_dummy_model()
    
    def create_dummy_model(self):
        """Create a dummy model for demo purposes"""
        # This would be replaced with your actual trained model
        return None
    
    def preprocess_image(self, image):
        """Preprocess medical image for ViT"""
        # Convert to RGB if needed
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize image
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def preprocess_text(self, text):
        """Preprocess clinical notes for BERT"""
        if not text or text.strip() == "":
            text = "No clinical notes provided."
        
        # Tokenize text
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='tf'
        )
        
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
    
    def predict(self, image, clinical_note):
        """Make prediction using multi-modal model"""
        try:
            # Preprocess inputs
            processed_image = self.preprocess_image(image)
            processed_text = self.preprocess_text(clinical_note)
            
            if self.model:
                # Make prediction
                predictions = self.model.predict({
                    'image_input': processed_image,
                    'input_ids': processed_text['input_ids'],
                    'attention_mask': processed_text['attention_mask']
                })
                
                multi_modal_preds = predictions['multi_modal_output'][0]
            else:
                # Demo mode - generate random predictions
                multi_modal_preds = np.random.uniform(0, 0.3, len(self.diseases))
                
                # Simulate some positive findings based on keywords
                note_lower = clinical_note.lower()
                if 'pneumonia' in note_lower:
                    multi_modal_preds[6] = 0.85  # Pneumonia
                if 'effusion' in note_lower:
                    multi_modal_preds[2] = 0.78  # Effusion
                if 'cardiomegaly' in note_lower:
                    multi_modal_preds[1] = 0.82  # Cardiomegaly
            
            return multi_modal_preds
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return np.zeros(len(self.diseases))
    
    def get_disease_explanations(self):
        """Return explanations for each disease"""
        explanations = {
            'Atelectasis': 'Collapse or closure of lung tissue',
            'Cardiomegaly': 'Enlarged heart shadow',
            'Effusion': 'Fluid in pleural space',
            'Infiltration': 'Inflammatory process in lung tissue',
            'Mass': 'Discrete spherical pulmonary lesion',
            'Nodule': 'Small round opacity in lung',
            'Pneumonia': 'Lung inflammation from infection',
            'Pneumothorax': 'Air in pleural space',
            'Consolidation': 'Airspace filling process',
            'Edema': 'Fluid accumulation in lungs',
            'Emphysema': 'Destruction of lung parenchyma',
            'Fibrosis': 'Scarring of lung tissue',
            'Pleural Thickening': 'Thickening of pleural lining',
            'Hernia': 'Diaphragmatic hernia'
        }
        return explanations