@echo off
echo Setting up Medical Multi-Modal Analysis Environment...

:: Create virtual environment
python -m venv medai_env
call medai_env\Scripts\activate.bat

:: Upgrade pip first
python -m pip install --upgrade pip

:: Install packages
pip install -r requirements.txt

:: Download pre-trained models
echo Downloading pre-trained models...
python -c "
from transformers import AutoTokenizer
import os

# Create model directory
os.makedirs('models', exist_ok=True)

# Download Bio_ClinicalBERT tokenizer
print('Downloading Bio_ClinicalBERT tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
tokenizer.save_pretrained('models/bio_clinical_bert_tokenizer')

print('Setup completed successfully!')
"

echo.
echo Activate the environment with: medai_env\Scripts\activate.bat
echo Run the app with: streamlit run app.py
pause