import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import streamlit as st

def create_probability_chart(predictions, diseases, threshold=0.5):
    """Create an interactive probability chart"""
    colors = ['red' if pred > threshold else 'blue' for pred in predictions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=diseases,
            y=predictions,
            marker_color=colors,
            text=[f'{p:.3f}' for p in predictions],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Disease Probability Scores',
        xaxis_title='Diseases',
        yaxis_title='Probability',
        yaxis_range=[0, 1],
        showlegend=False,
        height=500
    )
    
    # Add threshold line
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", 
                  annotation_text=f"Threshold: {threshold}")
    
    return fig

def create_radar_chart(predictions, diseases):
    """Create a radar chart for disease probabilities"""
    fig = go.Figure(data=go.Scatterpolar(
        r=np.append(predictions, predictions[0]),  # Close the radar
        theta=np.append(diseases, diseases[0]),
        fill='toself',
        line=dict(color='blue'),
        opacity=0.8
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="Disease Probability Radar",
        height=500
    )
    
    return fig

def create_comparison_plot(image_before, image_after, title1="Original", title2="Processed"):
    """Create side-by-side image comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(image_before, cmap='gray')
    ax1.set_title(title1)
    ax1.axis('off')
    
    ax2.imshow(image_after, cmap='gray')
    ax2.set_title(title2)
    ax2.axis('off')
    
    return fig

def highlight_keywords(text, keywords, color="yellow"):
    """Highlight medical keywords in clinical notes"""
    highlighted_text = text
    for keyword in keywords:
        highlighted_text = highlighted_text.replace(
            keyword, 
            f"<mark style='background-color: {color};'>{keyword}</mark>"
        )
    return highlighted_text