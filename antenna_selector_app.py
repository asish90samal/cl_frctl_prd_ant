import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# --------------------------------------------------------
# Page Configuration
# --------------------------------------------------------
st.set_page_config(
    page_title="Fractal Antenna Selector",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------
# Load trained model
# --------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")

model = load_model()

# --------------------------------------------------------
# Helper Functions
# --------------------------------------------------------

def calculate_rssi(env, distance, walls, metal, freq, antenna_type):
    """
    Simulate RSSI based on path loss model
    Antenna types: 0=Sierpinski, 1=Koch, 2=Monopole
    """
    Pt = 0.0  # Transmit power (dBm)
    d0 = 1.0
    
    # Path loss exponent based on environment
    n_base = {"indoor": 3.0, "urban": 2.7, "rural": 2.0}[env]
    n = n_base * (1.0 + (2400 - freq) / 2400 * 0.05)
    
    # Free space path loss at reference distance
    PLd0 = 30 + (2400 - freq) / 1000.0
    
    # Base RSSI calculation
    base_rssi = Pt - PLd0 - 10 * n * np.log10(max(distance, d0) / d0)
    
    # Wall penetration loss
    base_rssi -= walls * 3.5
    
    # Metal obstruction loss
    if metal:
        base_rssi -= 6.0
    
    # Antenna-specific offsets
    offsets = {
        "indoor": [0.5, 0.0, -1.0],
        "urban": [0.0, 0.7, -0.5],
        "rural": [-0.5, -0.2, 1.0]
    }
    
    ant_offset = offsets[env][antenna_type]
    
    # Frequency penalty
    freq_pen = 0.0
    if antenna_type == 2 and freq == 915:
        freq_pen += 0.6
    if antenna_type == 0 and freq == 2400:
        freq_pen += 0.2
    
    rssi = base_rssi + ant_offset + freq_pen
    
    return round(rssi, 2)

def get_prediction_confidence(model, df_input):
    """Get prediction probabilities for confidence scores"""
    try:
        # Get probability predictions
        proba = model.predict_proba(df_input)[0]
        return proba
    except:
        return None

def create_comparison_chart(env, distance, walls, metal, freq):
    """Create performance comparison chart for all antennas"""
    antenna_names = ["Sierpinski", "Koch", "Monopole"]
    rssi_values = []
    
    for i in range(3):
        rssi = calculate_rssi(env, distance, walls, metal, freq, i)
        rssi_values.append(rssi)
    
    # Create plotly bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=antenna_names,
            y=rssi_values,
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
            text=[f'{v:.1f} dBm' for v in rssi_values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Predicted RSSI for All Antenna Types",
        xaxis_title="Antenna Type",
        yaxis_title="RSSI (dBm)",
        height=400,
        template="plotly_white"
    )
    
    return fig

def create_3d_propagation_viz(env, distance, walls, metal, freq, best_antenna):
    """Create 3D visualization of signal propagation"""
    # Create distance range
    distances = np.linspace(1, min(distance * 1.5, 500), 50)
    
    # Calculate RSSI for each antenna across distances
    rssi_data = {
        'Sierpinski': [],
        'Koch': [],
        'Monopole': []
    }
    
    for d in distances:
        for i, name in enumerate(['Sierpinski', 'Koch', 'Monopole']):
            rssi = calculate_rssi(env, d, int(walls * d / distance), metal, freq, i)
            rssi_data[name].append(rssi)
    
    # Create 3D surface plot
    fig = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for i, (name, color) in enumerate(zip(['Sierpinski', 'Koch', 'Monopole'], colors)):
        fig.add_trace(go.Scatter(
            x=distances,
            y=rssi_data[name],
            mode='lines',
            name=name,
            line=dict(color=color, width=3 if i == best_antenna else 2),
            opacity=1.0 if i == best_antenna else 0.6
        ))
    
    fig.update_layout(
        title="Signal Strength vs Distance",
        xaxis_title="Distance (m)",
        yaxis_title="RSSI (dBm)",
        height=400,
        template="plotly_white",
        hovermode='x unified'
    )
    
    return fig

def generate_pdf_report(df_input, pred, antenna_name, rssi_values, confidence):
    """Generate PDF report"""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "Antenna Selection Report")
    
    # Date
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 70, f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Recommendation
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 110, "Recommended Antenna:")
    c.setFont("Helvetica", 12)
    c.drawString(70, height - 130, antenna_name)
    
    # Input Parameters
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 170, "Input Parameters:")
    c.setFont("Helvetica", 10)
    y_pos = height - 195
    for col, val in df_input.iloc[0].items():
        c.drawString(70, y_pos, f"{col}: {val}")
        y_pos -= 20
    
    # Performance Metrics
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_pos - 30, "Predicted RSSI Values:")
    c.setFont("Helvetica", 10)
    y_pos -= 55
    antenna_names = ["Sierpinski", "Koch", "Monopole"]
    for name, rssi in zip(antenna_names, rssi_values):
        marker = "✓" if name in antenna_name else " "
        c.drawString(70, y_pos, f"[{marker}] {name}: {rssi:.2f} dBm")
        y_pos -= 20
    
    # Confidence
    if confidence is not None:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_pos - 30, "Prediction Confidence:")
        c.setFont("Helvetica", 10)
        y_pos -= 55
        for i, (name, conf) in enumerate(zip(antenna_names, confidence)):
            c.drawString(70, y_pos, f"{name}: {conf*100:.1f}%")
            y_pos -= 20
    
    c.save()
    buffer.seek(0)
    return buffer

# --------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------
st.title("📡 AI-Powered Fractal Antenna Selector")
st.markdown("""
This intelligent system predicts the **optimal antenna type** based on environmental 
conditions using Machine Learning with advanced performance analytics.
""")

# Sidebar for inputs
with st.sidebar:
    st.header("⚙️ Configuration")
    
    env = st.selectbox(
        "Environment Type",
        ["urban", "rural", "indoor"],
        help="Select the deployment environment"
    )
    
    distance = st.number_input(
        "Distance (m)",
        min_value=1.0,
        max_value=1000.0,
        value=100.0,
        step=1.0,
        help="Distance between transmitter and receiver"
    )
    if distance > 500:
        st.warning("⚠️ Distance beyond 500 m – prediction may be less accurate.")
    
    walls = st.number_input(
        "Number of Walls",
        min_value=0,
        max_value=40,
        value=0,
        step=1,
        help="Number of walls between transmitter and receiver"
    )
    if walls > 10:
        st.warning("⚠️ Number of walls outside training range.")
    
    metal = st.selectbox(
        "Metal Obstructions?",
        ["No", "Yes"],
        help="Presence of metal obstacles"
    )
    metal_val = 1 if metal == "Yes" else 0
    
    freq = st.number_input(
        "Frequency (MHz)",
        min_value=0.0,
        max_value=6000.0,
        value=2400.0,
        step=10.0,
        help="Operating frequency"
    )
    if freq > 2400:
        st.warning("⚠️ Frequency beyond 2400 MHz.")
    
    st.markdown("---")
    predict_button = st.button("🔍 Analyze & Predict", use_container_width=True)

# --------------------------------------------------------
# Main Content
# --------------------------------------------------------

if predict_button:
    # Prepare input
    df_input = pd.DataFrame([{
        "env_type": env,
        "distance_m": distance,
        "num_walls": walls,
        "has_metal": metal_val,
        "frequency_mhz": freq
    }])
    
    # Make prediction
    pred = model.predict(df_input)[0]
    confidence = get_prediction_confidence(model, df_input)
    
    antenna_map = {
        0: ("Sierpinski Fractal Antenna", "designs/sierpinski.png"),
        1: ("Koch Fractal Antenna", "designs/koch.png"),
        2: ("Monopole Antenna", "designs/monopole.png")
    }
    
    antenna_name, antenna_image = antenna_map[pred]
    
    # Calculate RSSI for all antennas
    rssi_values = [
        calculate_rssi(env, distance, walls, metal_val, freq, 0),
        calculate_rssi(env, distance, walls, metal_val, freq, 1),
        calculate_rssi(env, distance, walls, metal_val, freq, 2)
    ]
    
    # Display Results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.success(f"### ✅ Recommended: **{antenna_name}**")
        
        # Improvement #3: Confidence Scores
        if confidence is not None:
            st.markdown("#### 📊 Prediction Confidence")
            conf_df = pd.DataFrame({
                'Antenna': ['Sierpinski', 'Koch', 'Monopole'],
                'Confidence': [f"{c*100:.1f}%" for c in confidence]
            })
            st.dataframe(conf_df, use_container_width=True, hide_index=True)
            
            # Visual confidence bars
            for i, (name, conf) in enumerate(zip(['Sierpinski', 'Koch', 'Monopole'], confidence)):
                st.progress(float(conf), text=f"{name}: {conf*100:.1f}%")
        
        # Improvement #5: Real-time RSSI Estimation
        st.markdown("#### 📶 Predicted Signal Strength")
        st.metric(
            label=antenna_name,
            value=f"{rssi_values[pred]:.2f} dBm",
            delta=f"{rssi_values[pred] - np.mean(rssi_values):.2f} dBm vs avg",
            delta_color="normal"
        )
        
        # Display antenna image
        if os.path.exists(antenna_image):
            st.image(antenna_image, caption=antenna_name, use_container_width=True)
    
    with col2:
        # Improvement #2: Performance Comparison Chart
        st.markdown("#### 📊 Performance Comparison")
        comparison_fig = create_comparison_chart(env, distance, walls, metal_val, freq)
        st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Display all RSSI values
        rssi_df = pd.DataFrame({
            'Antenna': ['Sierpinski', 'Koch', 'Monopole'],
            'RSSI (dBm)': rssi_values,
            'Rank': [1, 2, 3]
        })
        rssi_df = rssi_df.sort_values('RSSI (dBm)', ascending=False).reset_index(drop=True)
        rssi_df['Rank'] = range(1, 4)
        st.dataframe(rssi_df, use_container_width=True, hide_index=True)
    
    # Improvement #6: 3D Visualization
    st.markdown("---")
    st.markdown("### 📈 Signal Propagation Analysis")
    propagation_fig = create_3d_propagation_viz(env, distance, walls, metal_val, freq, pred)
    st.plotly_chart(propagation_fig, use_container_width=True)
    
    # Input Summary
    st.markdown("---")
    st.markdown("### 📋 Configuration Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Environment", env)
        st.metric("Distance", f"{distance} m")
    with col2:
        st.metric("Walls", int(walls))
        st.metric("Metal", metal)
    with col3:
        st.metric("Frequency", f"{freq} MHz")
    
    # Improvement #7: Export Functionality
    st.markdown("---")
    st.markdown("### 📥 Export Report")
    
    col1, col2 = st.columns(2)
    with col1:
        # Download as CSV
        csv_data = df_input.copy()
        csv_data['recommended_antenna'] = antenna_name
        csv_data['predicted_rssi'] = rssi_values[pred]
        csv = csv_data.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV Report",
            data=csv,
            file_name=f"antenna_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Download as PDF
        pdf_buffer = generate_pdf_report(df_input, pred, antenna_name, rssi_values, confidence)
        st.download_button(
            label="📑 Download PDF Report",
            data=pdf_buffer,
            file_name=f"antenna_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

else:
    # Landing page content
    st.info("👈 Configure parameters in the sidebar and click **Analyze & Predict** to get started!")
    
    # Show antenna designs
    st.markdown("### 📡 Available Antenna Types")
    col1, col2, col3 = st.columns(3)
    
    designs = [
        ("Sierpinski Fractal", "designs/sierpinski.png", "Excellent for indoor multi-path environments"),
        ("Koch Fractal", "designs/koch.png", "Optimal for urban deployments with obstacles"),
        ("Monopole", "designs/monopole.png", "Best for open rural areas")
    ]
    
    for col, (name, img, desc) in zip([col1, col2, col3], designs):
        with col:
            st.markdown(f"**{name}**")
            if os.path.exists(img):
                st.image(img, use_container_width=True)
            st.caption(desc)

# --------------------------------------------------------
# Footer
# --------------------------------------------------------
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🤖 Powered by Machine Learning")
with col2:
    st.caption("📡 Fractal Antenna Research")
with col3:
    st.caption("🚀 Deployed on Streamlit Cloud")