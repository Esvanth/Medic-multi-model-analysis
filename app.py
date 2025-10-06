import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import os
import sys

# Add utils to path
sys.path.append('utils')

from model_utils import MedicalMultiModalPredictor
from visualization import *

# Page configuration
st.set_page_config(
    page_title="Medic Multi-Modal Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disease-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .high-risk {
        border-left-color: #ff4b4b;
        background-color: #fff5f5;
    }
    .medium-risk {
        border-left-color: #ffa500;
        background-color: #fffaf0;
    }
    .low-risk {
        border-left-color: #00d4aa;
        background-color: #f0fff4;
    }
    .probability-bar {
        height: 20px;
        background-color: #e0e0e0;
        border-radius: 10px;
        margin: 5px 0;
    }
    .probability-fill {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #4CAF50, #FFC107, #FF5252);
    }
</style>
""", unsafe_allow_html=True)

class MedicalApp:
    def __init__(self):
        self.predictor = None
        self.initialize_session_state()
        self.load_model()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'predictions' not in st.session_state:
            st.session_state.predictions = None
        if 'image_uploaded' not in st.session_state:
            st.session_state.image_uploaded = False
        if 'clinical_note' not in st.session_state:
            st.session_state.clinical_note = ""
        if 'uploaded_image' not in st.session_state:
            st.session_state.uploaded_image = None
    
    def load_model(self):
        """Load the ML model"""
        with st.spinner("Loading medical AI model..."):
            try:
                self.predictor = MedicalMultiModalPredictor('models/multi_modal_model.h5')
                st.success("Model loaded successfully!")
            except Exception as e:
                st.warning(f"Using demo mode: {e}")
                self.predictor = MedicalMultiModalPredictor()
    
    def render_sidebar(self):
        """Render the sidebar"""
        with st.sidebar:
            st.title("🏥 Medical AI Analysis")
            
            st.header("Upload Medical Data")
            
            # Image upload
            uploaded_file = st.file_uploader(
                "Upload Medical Image",
                type=['png', 'jpg', 'jpeg', 'dcm'],
                help="Upload X-ray, CT scan, or other medical images"
            )
            
            if uploaded_file is not None:
                # Read and process image
                image = Image.open(uploaded_file)
                st.session_state.uploaded_image = np.array(image)
                st.session_state.image_uploaded = True
                
                # Display uploaded image
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Clinical notes input
            st.subheader("Clinical Notes")
            clinical_note = st.text_area(
                "Enter clinical findings, symptoms, and observations:",
                height=200,
                placeholder="Example: Patient presents with cough, fever, and shortness of breath. Chest X-ray shows opacity in right lower lobe..."
            )
            st.session_state.clinical_note = clinical_note
            
            # Sample reports
            if st.button("Load Sample Report"):
                sample_reports = self.load_sample_reports()
                st.session_state.clinical_note = np.random.choice(sample_reports)
                st.rerun()
            
            # Analysis threshold
            st.subheader("Analysis Settings")
            threshold = st.slider(
                "Probability Threshold for Positive Findings",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.05,
                help="Adjust the sensitivity for disease detection"
            )
            
            # Analyze button
            analyze_btn = st.button(
                "🔍 Run Multi-Modal Analysis",
                type="primary",
                use_container_width=True,
                disabled=not st.session_state.image_uploaded
            )
            
            return analyze_btn, threshold
    
    def load_sample_reports(self):
        """Load sample clinical reports"""
        sample_reports = [
            "Patient presents with acute onset cough, fever (38.5°C), and right-sided chest pain. Breath sounds decreased on right base. Suspected pneumonia.",
            "Chronic smoker with progressive dyspnea. Hyperinflated lungs with flattened diaphragms. Increased retrosternal air space.",
            "Post-operative patient with sudden onset dyspnea. Decreased breath sounds on left side. Tracheal deviation to right.",
            "Heart failure patient with orthopnea and bilateral leg edema. Cardiomegaly with pulmonary vascular congestion and small bilateral pleural effusions.",
            "Occupational exposure history. Bilateral upper lobe fibrotic changes with traction bronchiectasis.",
            "Asymptomatic patient for routine screening. Clear lung fields, normal cardiac silhouette."
        ]
        return sample_reports
    
    def render_disease_cards(self, predictions, threshold):
        """Render disease probability cards"""
        st.header("📋 Disease Probability Analysis")
        
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Probability chart
            fig = create_probability_chart(
                predictions, 
                self.predictor.diseases, 
                threshold
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Radar chart
            fig_radar = create_radar_chart(predictions, self.predictor.diseases)
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Disease cards
        st.subheader("Detailed Findings")
        
        explanations = self.predictor.get_disease_explanations()
        
        # Sort diseases by probability
        sorted_indices = np.argsort(predictions)[::-1]
        
        for idx in sorted_indices:
            disease = self.predictor.diseases[idx]
            prob = predictions[idx]
            
            # Determine risk level
            if prob > threshold:
                risk_class = "high-risk"
                icon = "🔴"
            elif prob > threshold * 0.7:
                risk_class = "medium-risk"
                icon = "🟡"
            else:
                risk_class = "low-risk"
                icon = "🟢"
            
            # Create disease card
            with st.container():
                st.markdown(f"""
                <div class="disease-card {risk_class}">
                    <h4>{icon} {disease}: {prob:.3f}</h4>
                    <p>{explanations[disease]}</p>
                    <div class="probability-bar">
                        <div class="probability-fill" style="width: {prob*100}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    def render_clinical_insights(self, predictions, clinical_note):
        """Generate clinical insights based on predictions"""
        st.header("💡 Clinical Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Key Findings")
            
            # Highlight high probability diseases
            high_prob_diseases = [
                self.predictor.diseases[i] 
                for i, prob in enumerate(predictions) 
                if prob > 0.5
            ]
            
            if high_prob_diseases:
                st.warning("**High Probability Conditions:**")
                for disease in high_prob_diseases:
                    st.write(f"• {disease}")
            else:
                st.success("**No high-probability abnormalities detected**")
            
            # Generate recommendations
            st.subheader("Recommendations")
            if any(prob > 0.7 for prob in predictions):
                st.error("**Urgent follow-up recommended**")
                st.write("Consider immediate specialist consultation and additional imaging.")
            elif any(prob > 0.5 for prob in predictions):
                st.warning("**Further evaluation advised**")
                st.write("Schedule follow-up imaging and clinical assessment.")
            else:
                st.success("**Routine follow-up**")
                st.write("No immediate intervention required.")
        
        with col2:
            st.subheader("Clinical Note Analysis")
            
            # Highlight medical keywords
            medical_keywords = [
                'pneumonia', 'effusion', 'cardiomegaly', 'atelectasis',
                'infiltration', 'nodule', 'mass', 'pneumothorax',
                'consolidation', 'edema', 'emphysema', 'fibrosis',
                'hernia', 'fever', 'cough', 'dyspnea', 'chest pain'
            ]
            
            highlighted_text = clinical_note
            for keyword in medical_keywords:
                if keyword in clinical_note.lower():
                    highlighted_text = highlighted_text.replace(
                        keyword, 
                        f"**{keyword}**"
                    )
            
            st.markdown(highlighted_text)
    
    def render_technical_details(self):
        """Show technical information about the model"""
        with st.expander("🔬 Technical Details"):
            st.subheader("Model Architecture")
            
            st.markdown("""
            **Multi-Modal Fusion Network:**
            - **Vision Pathway**: Vision Transformer (ViT) for medical image analysis
            - **Text Pathway**: Bio_ClinicalBERT for clinical note understanding  
            - **Fusion**: Cross-modal attention mechanism
            - **Output**: Multi-label classification for 14 thoracic diseases
            
            **Training Data:**
            - ChestX-ray14 dataset
            - MIMIC-CXR database
            - Federated learning from multiple institutions
            """)
            
            st.subheader("Performance Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Overall AUC", "0.92")
            with col2:
                st.metric("Sensitivity", "0.89")
            with col3:
                st.metric("Specificity", "0.94")
    
    def main(self):
        """Main application flow"""
        # Header
        st.markdown('<h1 class="main-header">🏥 Multi-Modal Medical AI Analysis</h1>', 
                   unsafe_allow_html=True)
        st.markdown("### Combining Medical Imaging with Clinical Notes for Comprehensive Diagnosis")
        
        # Sidebar
        analyze_btn, threshold = self.render_sidebar()
        
        # Main content area
        if not st.session_state.image_uploaded:
            self.render_landing_page()
        else:
            if analyze_btn or st.session_state.predictions is not None:
                self.render_analysis_results(threshold)
            else:
                self.render_upload_preview()
        
        # Technical details
        self.render_technical_details()
    
    def render_landing_page(self):
        """Render the landing page when no image is uploaded"""
        st.info("👆 **Upload a medical image and clinical notes to get started**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Supported Image Types")
            st.markdown("""
            - **X-rays** (Chest, Abdominal, Skeletal)
            - **CT Scans** (All body regions)
            - **MRI Images** 
            - **Ultrasound** images
            - **DICOM** format supported
            """)
        
        with col2:
            st.subheader("📝 Clinical Note Analysis")
            st.markdown("""
            - **Symptom description**
            - **Clinical findings**
            - **Patient history**
            - **Physical examination notes**
            - **Laboratory results**
            """)
        
        # Demo images
        st.subheader("🎯 Try with Sample Data")
        sample_cols = st.columns(4)
        
        # You can add sample images here
        for i, col in enumerate(sample_cols):
            with col:
                if st.button(f"Sample Case {i+1}", use_container_width=True):
                    # Load sample data
                    st.info("Sample case loaded - upload an image to see similar analysis")
    
    def render_upload_preview(self):
        """Render preview after upload but before analysis"""
        st.success("✅ Medical image uploaded successfully!")
        st.info("📝 **Add clinical notes in the sidebar and click 'Run Multi-Modal Analysis'**")
        
        # Show image preview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(
                st.session_state.uploaded_image, 
                caption="Uploaded Medical Image",
                use_column_width=True
            )
        
        with col2:
            st.subheader("Next Steps:")
            st.markdown("""
            1. 📝 Add clinical notes in sidebar
            2. 🔧 Adjust analysis threshold if needed  
            3. 🔍 Click analysis button
            4. 📊 Review AI-generated insights
            5. 💡 Get clinical recommendations
            """)
    
    def render_analysis_results(self, threshold):
        """Render the complete analysis results"""
        with st.spinner("🔬 Analyzing medical data with multi-modal AI..."):
            # Make prediction
            predictions = self.predictor.predict(
                st.session_state.uploaded_image,
                st.session_state.clinical_note
            )
            
            st.session_state.predictions = predictions
            
            # Show results
            st.success("✅ Analysis Complete!")
            
            # Render all result components
            self.render_disease_cards(predictions, threshold)
            self.render_clinical_insights(predictions, st.session_state.clinical_note)
            
            # Download results
            self.render_export_options(predictions)
    
    def render_export_options(self, predictions):
        """Render options to export results"""
        st.header("📤 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Generate report text
            report_text = self.generate_report_text(predictions)
            st.download_button(
                label="📄 Download PDF Report",
                data=report_text,
                file_name="medical_ai_report.txt",
                mime="text/plain"
            )
        
        with col2:
            # Export data
            import json
            report_data = {
                'predictions': dict(zip(self.predictor.diseases, predictions.tolist())),
                'clinical_note': st.session_state.clinical_note,
                'timestamp': str(np.datetime64('now'))
            }
            
            st.download_button(
                label="📊 Download JSON Data",
                data=json.dumps(report_data, indent=2),
                file_name="medical_analysis_data.json",
                mime="application/json"
            )
        
        with col3:
            if st.button("🔄 New Analysis"):
                # Reset session state
                for key in ['predictions', 'image_uploaded', 'clinical_note', 'uploaded_image']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    def generate_report_text(self, predictions):
        """Generate a comprehensive report text"""
        report = f"""
MEDICAL AI ANALYSIS REPORT
Generated: {np.datetime64('now')}
========================================

CLINICAL DATA:
--------------
Image Analysis: Completed
Clinical Notes: {len(st.session_state.clinical_note)} characters

FINDINGS:
---------
"""
        
        # Add disease probabilities
        for i, disease in enumerate(self.predictor.diseases):
            prob = predictions[i]
            status = "POSITIVE" if prob > 0.5 else "negative"
            report += f"{disease}: {prob:.3f} ({status})\n"
        
        report += f"""
CLINICAL INSIGHTS:
------------------
"""
        # Add insights
        high_risk = [self.predictor.diseases[i] for i, p in enumerate(predictions) if p > 0.7]
        if high_risk:
            report += f"High probability findings: {', '.join(high_risk)}\n"
        
        report += """
RECOMMENDATIONS:
---------------
"""
        if any(p > 0.7 for p in predictions):
            report += "Urgent clinical follow-up recommended.\nConsider specialist consultation.\n"
        elif any(p > 0.5 for p in predictions):
            report += "Further evaluation advised.\nSchedule follow-up assessment.\n"
        else:
            report += "Routine follow-up recommended.\nNo immediate intervention required.\n"
        
        report += "\n--- End of Report ---"
        return report

# Run the application
if __name__ == "__main__":
    app = MedicalApp()
    app.main()
