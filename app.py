import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Animal Classifier",
    page_icon="🦁",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🦁 Animal Species Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload an image to identify the animal species</p>', unsafe_allow_html=True)

# Class names (90 animal classes)
@st.cache_data
def get_class_names():
    return [
        'antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 
        'butterfly', 'cat', 'caterpillar', 'chimpanzee', 'cockroach', 'cow', 
        'coyote', 'crab', 'crow', 'deer', 'dog', 'dolphin', 'donkey', 'dragonfly', 
        'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 'goat', 'goldfish', 
        'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog', 
        'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish', 
        'kangaroo', 'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 
        'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'orangutan', 'otter', 
        'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 'penguin', 
        'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer', 
        'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 
        'sparrow', 'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 
        'turtle', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra'
    ]

@st.cache_resource
def load_model():
    try:
        st.sidebar.info("**Animal Identifier**")
        
        # Check if weight files exist
        if not os.path.exists("dense_weights.npy") or not os.path.exists("dense_bias.npy"):
            st.error("❌ Weight files not found! Please run extraction first.")
            st.info("Required files: dense_weights.npy, dense_bias.npy")
            return None
        
        # Load the saved dense weights
        dense_weights = np.load("dense_weights.npy")
        dense_bias = np.load("dense_bias.npy")
        
        st.sidebar.info(f"Locked and Loaded - Shape: {dense_weights.shape}")
        
        # Build model with pre-trained hub layer
        hub_layer = hub.KerasLayer(
            "https://tfhub.dev/google/efficientnet/b0/feature-vector/1",
            trainable=False
        )
        
        # Create the model
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = hub_layer(inputs)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(90, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Set the trained weights
        model.layers[-1].set_weights([dense_weights, dense_bias])
        
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Test with random input
        test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        test_pred = model.predict(test_input, verbose=0)[0]
        st.sidebar.success(f"**Model is ready!** {test_pred.max():.1%}")
        
        return model
        
    except Exception as e:
        st.error(f"❌ Error building model: {e}")
        return None

# Load model
model = load_model()
class_names = get_class_names()

# Sidebar
with st.sidebar:
    st.header("📋 About")
    st.info("""
**90 Animal Species Classifier** 🤖
EfficientNetB0 + Custom Training | 90 Classes
""")
    
    st.header("📊 Dataset Info")
    st.metric("Total Classes", len(class_names))
    
    if model is not None:
        st.success("AI is ready!")

# Main content
if model is not None:
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Upload a clear image of an animal"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.write(f"**Image size:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**Format:** {image.format}")
        
        with col2:
            if st.button("🔍 Classify Animal", type="primary", use_container_width=True):
                with st.spinner("🔬 Analyzing image..."):
                    # Preprocess image
                    img = image.resize((224, 224))
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
                    
                    # Make prediction
                    predictions = model.predict(img_array, verbose=0)[0]
                    
                    # Get top 5 predictions
                    top_5_idx = np.argsort(predictions)[-5:][::-1]
                    top_5_conf = predictions[top_5_idx]
                    top_5_classes = [class_names[i] for i in top_5_idx]
                    
                    # Display results
                    st.subheader("📊 Top 5 Predictions")
                    
                    for i, (cls, conf) in enumerate(zip(top_5_classes, top_5_conf)):
                        # Color code based on confidence
                        if conf > 0.8:
                            color = "🟢"
                        elif conf > 0.5:
                            color = "🟡"
                        else:
                            color = "🔴"
                        
                        st.write(f"{color} **{i+1}. {cls}**")
                        st.progress(float(conf))
                        st.write(f"Confidence: **{conf:.2%}**")
                        if i < 4:
                            st.write("---")
                    
                    # Show top prediction prominently
                    st.markdown("---")
                    if top_5_conf[0] > 0.8:
                        st.success(f"🎯 **Prediction: {top_5_classes[0]}** ({top_5_conf[0]:.2%})")
                    elif top_5_conf[0] > 0.5:
                        st.warning(f"⚠️ **Maybe: {top_5_classes[0]}** ({top_5_conf[0]:.2%})")
                    else:
                        st.error(f"❓ **Unsure: {top_5_classes[0]}** ({top_5_conf[0]:.2%})")

else:
    st.error("❌ Model failed to load")
    st.write("📁 Files in current directory:")
    for file in os.listdir():
        if file.endswith(('.npy', '.pkl', '.txt')):
            size = os.path.getsize(file) / (1024 * 1024)
            st.write(f"- {file}: {size:.2f} MB")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        Made using Streamlit and TensorFlow | Recognizes 90 Animal Species
    </div>
    """,
    unsafe_allow_html=True
)