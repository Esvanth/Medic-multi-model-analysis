import sys
import subprocess
def test_imports():
    """Test if all required packages can be imported"""
    packages = [
        'streamlit',
        'tensorflow',
        'transformers',
        'torch',
        'cv2',
        'PIL',
        'pandas',
        'numpy',
        'sklearn',
        'matplotlib',
        'seaborn',
        'plotly'
    ]
    print("Testing package imports...")
    for package in packages:
        try:
            if package == 'PIL':
                __import__('PIL.Image')
            elif package == 'cv2':
                __import__('cv2')
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
    print("\nTesting model loading...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        print("✅ Bio_ClinicalBERT tokenizer")
    except Exception as e:
        print(f"❌ Tokenizer: {e}")
if __name__ == "__main__":
    test_imports()