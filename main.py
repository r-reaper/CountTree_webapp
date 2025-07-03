import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
import tempfile
import os
import base64 # Import base64 for image encoding (kept for potential future use or debugging, but not directly used for st_canvas background_image in this fix)
from ultralytics import YOLO # Import YOLO class
from streamlit_drawable_canvas import st_canvas # Import the drawable canvas component

# --- Function to load YOLO model ---
@st.cache_resource
def load_yolo_model(model_path='best.pt'):
    """
    Loads the YOLO model from the specified path.
    Uses st.cache_resource to cache the model, preventing reloads on every interaction.
    """
    try:
        model = YOLO(model_path)
        st.success(f"YOLO model '{model_path}' loaded successfully. Device: {model.device}")
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        st.error("**Please check:**")
        st.error(f"1. The file `{model_path}` (best.pt) exists and the path is correct in the sidebar.")
        st.error("2. Your `best.pt` file is a model trained from the Ultralytics Framework (YOLOv8/YOLOv11).")
        st.stop() # Stop Streamlit execution if model fails to load

# --- Function to process image for YOLO (handles dtype and channels) ---
def process_image_for_yolo(image_np):
    """
    Processes the input image NumPy array to ensure it's in RGB and uint8 format,
    suitable for YOLO inference and display in Streamlit.
    """
    # Ensure the image has at least 3 dimensions (H, W, C) for color processing
    if image_np.ndim == 2: # Grayscale image (H, W)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4: # RGBA image (H, W, 4)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    elif image_np.shape[2] > 4: # Handle cases with more than 4 channels (e.g., multispectral)
        # For display and standard YOLO, we typically only need 3 channels (RGB approximation)
        image_np = image_np[:, :, :3]

    # Convert to uint8 and scale if necessary for display and YOLO
    if image_np.dtype != np.uint8:
        if np.issubdtype(image_np.dtype, np.integer) and np.max(image_np) > 255:
            # If it's a higher bit depth integer image (e.g., 16-bit), scale it down to 8-bit (0-255)
            max_val = np.max(image_np)
            if max_val > 0:
                image_np = (image_np / max_val * 255).astype(np.uint8)
            else: # Image is all black, just convert to uint8
                image_np = image_np.astype(np.uint8)
        elif np.issubdtype(image_np.dtype, np.floating):
            # If it's float (e.g., 0.0-1.0), scale to 0-255 and convert to uint8
            image_np = (image_np * 255).astype(np.uint8)
        else: # Just convert if it's already in 0-255 range but wrong dtype (e.g., int8)
            image_np = image_np.astype(np.uint8)

    return image_np

# --- Function to detect trees in ROI by tiling ---
def detect_trees_in_roi(model, roi_image_np, roi_offset_x, roi_offset_y, TILE_SIZE=640, OVERLAP=100, conf_threshold=0.25, iou_threshold=0.7):
    """
    Detects trees within a specified Region of Interest (ROI) by tiling the ROI
    and running YOLO inference on each tile.
    """
    detected_trees = []
    roi_height, roi_width, _ = roi_image_np.shape

    # Create a copy of the ROI image for drawing bounding boxes and centers
    display_roi_image = roi_image_np.copy()

    # Iterate through the ROI, creating overlapping tiles
    for y_start_tile in range(0, roi_height, TILE_SIZE - OVERLAP):
        for x_start_tile in range(0, roi_width, TILE_SIZE - OVERLAP):
            x_end_tile = min(x_start_tile + TILE_SIZE, roi_width)
            y_end_tile = min(y_start_tile + TILE_SIZE, roi_height)

            # Adjust start coordinates if the tile goes beyond bounds (for the last tiles)
            if x_end_tile - x_start_tile < TILE_SIZE and x_start_tile != 0:
                x_start_tile = max(0, x_end_tile - TILE_SIZE)
            if y_end_tile - y_start_tile < TILE_SIZE and y_start_tile != 0:
                y_start_tile = max(0, y_end_tile - TILE_SIZE)
            
            # Ensure start coordinates are not negative
            x_start_tile = max(0, x_start_tile)
            y_start_tile = max(0, y_start_tile)

            # Extract the tile from the ROI image
            tile = roi_image_np[y_start_tile:y_end_tile, x_start_tile:x_end_tile]

            # Pad tile if its dimensions are less than TILE_SIZE to match model input size
            padded_tile = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
            padded_tile[:tile.shape[0], :tile.shape[1], :] = tile
            
            # Perform inference using the YOLO model object
            # Pass confidence_threshold and iou_threshold from UI
            results = model(padded_tile, imgsz=TILE_SIZE, conf=conf_threshold, iou=iou_threshold)

            # Process detection results
            for r in results: # Iterate over results (one per image if not batching)
                boxes = r.boxes.xyxy.cpu().numpy() # Bounding box coordinates (xmin, ymin, xmax, ymax)
                confidences = r.boxes.conf.cpu().numpy() # Confidence scores
                classes = r.boxes.cls.cpu().numpy() # Class IDs

                for i in range(len(boxes)):
                    x_min_rel, y_min_rel, x_max_rel, y_max_rel = boxes[i]
                    conf = confidences[i]
                    cls_id = int(classes[i])
                    class_name = model.names[cls_id] # Get class name from model

                    # Coordinates from padded_tile are already relative to the padded input
                    x_min_on_tile = x_min_rel
                    y_min_on_tile = y_min_rel
                    x_max_on_tile = x_max_rel
                    y_max_on_tile = y_max_rel
                    
                    # Convert coordinates from tile to ROI coordinate system
                    x_min_roi = x_start_tile + x_min_on_tile
                    y_min_roi = y_start_tile + y_min_on_tile
                    x_max_roi = x_start_tile + x_max_on_tile
                    y_max_roi = y_start_tile + y_max_on_tile

                    # Convert coordinates from ROI to original image coordinate system
                    x_min_orig = roi_offset_x + x_min_roi
                    y_min_orig = roi_offset_y + y_min_roi
                    x_max_orig = roi_offset_x + x_max_roi
                    y_max_orig = roi_offset_y + y_max_roi

                    # Calculate center coordinates (pixel) of the detected object
                    center_x_pixel = (x_min_orig + x_max_orig) / 2
                    center_y_pixel = (y_min_orig + y_max_orig) / 2

                    detected_trees.append({
                        'bbox_pixel': [int(x_min_orig), int(y_min_orig), int(x_max_orig), int(y_max_orig)],
                        'confidence': conf,
                        'class': class_name,
                        'center_coords_pixel': (center_x_pixel, center_y_pixel)
                    })
                    
                    # Draw bounding box and center on the display_roi_image for visualization
                    cv2.rectangle(display_roi_image, (int(x_min_roi), int(y_min_roi)), (int(x_max_roi), int(y_max_roi)), (0, 255, 0), 2) # Green box
                    cv2.circle(display_roi_image, (int((x_min_roi + x_max_roi) / 2), int((y_min_roi + y_max_roi) / 2)), 3, (0, 0, 255), -1) # Red dot

    return detected_trees, display_roi_image

# --- Streamlit UI Configuration ---
st.set_page_config(layout="wide") # Use wide layout for better display of large images
st.title("🌲 Orthophoto Tree Detection with YOLO")

st.markdown("""
Upload your Orthophoto (.tiff, .png) image, then select the area of interest to detect trees.
""")

# --- Sidebar for file upload and model settings ---
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Select Orthophoto File (.tiff, .png)", type=["tiff", "tif", "png"])

    st.header("Model Settings")
    # Input for model path (default to common YOLOv11 model name)
    model_path_input = st.text_input("Path to best.pt model file:", "best.pt")
    
    # Load the YOLO model (cached)
    model = load_yolo_model(model_path_input)

    st.markdown("---") # Separator
    st.header("Detection Settings")
    # Slider for Confidence Threshold
    confidence_threshold = st.slider(
        "Set Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25, # Default value
        step=0.05,
        help="Minimum confidence score for a detection to be considered valid (0.0 - 1.0)"
    )
    # Slider for IoU Threshold (for Non-Maximum Suppression)
    iou_threshold = st.slider(
        "Set IoU Threshold (NMS)",
        min_value=0.0,
        max_value=1.0,
        value=0.7, # Default value
        step=0.05,
        help="Maximum Intersection Over Union (IoU) for non-maximum suppression (NMS) to filter overlapping bounding boxes."
    )

# --- Main content area for image display and results ---
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    
    temp_image_path = None # Initialize to None for finally block
    try:
        # Save the uploaded file to a temporary location to be read by PIL/cv2
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.type.split('/')[-1]}") as tmp_file:
            tmp_file.write(bytes_data)
            temp_image_path = tmp_file.name

        # Open image using PIL (handles both TIFF and PNG)
        original_image_pil = Image.open(temp_image_path)
        original_image_np = np.array(original_image_pil)
        
        # Process image for YOLO (ensure RGB and uint8)
        original_image_np = process_image_for_yolo(original_image_np)

        original_height, original_width, _ = original_image_np.shape
        st.write(f"Orthophoto Image Size: {original_width}x{original_height} px")

        st.subheader("Select Region of Interest (ROI) by Drawing")
        st.info("Draw a rectangle on the image below to define your ROI. Only the first drawn rectangle will be used.")

        # --- Display downscaled image on canvas for ROI selection ---
        display_scale = 0.2 # Scale down for display, adjust as needed
        # Ensure dimensions are at least 1 to avoid errors for tiny images
        canvas_width = max(1, int(original_width * display_scale))
        canvas_height = max(1, int(original_height * display_scale))
        
        # Ensure image is uint8 before resizing for display
        if original_image_np.dtype != np.uint8:
            st.error("Error: Image dtype is not uint8 after processing. Cannot resize for display.")
            st.stop() # Stop execution if image is not in expected format

        display_image_for_canvas = cv2.resize(original_image_np, (canvas_width, canvas_height))
        
        # FIX: Pass PIL Image directly to background_image
        # The previous error "AttributeError: 'str' object has no attribute 'height'"
        # suggests that st_canvas expects an image object (like PIL Image) for background_image,
        # not a base64 string directly, even though its docs might imply it.
        # Converting to PIL Image from numpy array is the most robust way.
        background_image_pil = Image.fromarray(display_image_for_canvas)

        # Use st_canvas for drawing ROI
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Orange translucent fill
            stroke_width=2,
            stroke_color="rgba(255, 165, 0, 1)", # Orange stroke
            background_image=background_image_pil, # Pass PIL Image object directly here
            height=canvas_height,
            width=canvas_width,
            drawing_mode="rect", # Allow drawing rectangles
            key="canvas",
        )

        roi_x_min, roi_y_min, roi_x_max, roi_y_max = 0, 0, original_width, original_height # Default to full image

        if canvas_result.json_data is not None:
            objects = canvas_result.json_data.get("objects", [])
            if objects:
                # Get the last drawn object (assuming it's a rectangle)
                last_object = objects[-1]
                if last_object["type"] == "rect":
                    # Coordinates from canvas are relative to the displayed (scaled) image
                    x_on_canvas = last_object["left"]
                    y_on_canvas = last_object["top"]
                    width_on_canvas = last_object["width"]
                    height_on_canvas = last_object["height"]

                    # Scale coordinates back to original image dimensions
                    roi_x_min = int(x_on_canvas / display_scale)
                    roi_y_min = int(y_on_canvas / display_scale)
                    roi_x_max = int((x_on_canvas + width_on_canvas) / display_scale)
                    roi_y_max = int((y_on_canvas + height_on_canvas) / display_scale)

                    # Ensure coordinates are within original image bounds
                    roi_x_min = max(0, roi_x_min)
                    roi_y_min = max(0, roi_y_min)
                    roi_x_max = min(original_width, roi_x_max)
                    roi_y_max = min(original_height, roi_y_max)

                    st.success(f"Selected ROI: ({roi_x_min},{roi_y_min}) - ({roi_x_max},{roi_y_max})")
                else:
                    st.warning("Please draw a rectangle to define the ROI.")
            else:
                st.info("Draw a rectangle on the image above to define the ROI.")
        else:
            st.info("Draw a rectangle on the image above to define the ROI.")

        # --- Crop ROI image and display preview (using drawn ROI or default) ---
        # Ensure ROI coordinates are within image bounds before cropping
        roi_x_min = max(0, roi_x_min)
        roi_y_min = max(0, roi_y_min)
        roi_x_max = min(original_width, roi_x_max)
        roi_y_max = min(original_height, roi_y_max)

        # Check if a valid ROI was selected, otherwise use full image
        if roi_x_max > roi_x_min and roi_y_max > roi_y_min:
            roi_image = original_image_np[roi_y_min:roi_y_max, roi_x_min:roi_x_max].copy()
            st.image(roi_image, caption=f"Selected ROI Image ({roi_image.shape[1]}x{roi_image.shape[0]} px)", use_column_width=True)
        else:
            st.warning("No valid ROI drawn or selected. Processing the full image.")
            roi_image = original_image_np.copy()
            roi_x_min, roi_y_min = 0, 0 # Reset offsets for full image processing


        # --- Button to start detection ---
        if st.button("Start Tree Detection"):
            with st.spinner("Detecting trees... Please wait."):
                # Call the detection function with selected thresholds
                detected_trees_list, roi_display_image_with_bboxes = detect_trees_in_roi(
                    model,
                    roi_image,
                    roi_offset_x=roi_x_min,
                    roi_offset_y=roi_y_min,
                    conf_threshold=confidence_threshold, # Pass confidence from slider
                    iou_threshold=iou_threshold # Pass IoU from slider
                )

            st.subheader("Tree Detection Results")
            st.write(f"Total trees detected: **{len(detected_trees_list)}** trees")

            # Display ROI image with bounding boxes
            st.image(roi_display_image_with_bboxes, caption="ROI Image with Detections", use_column_width=True)

            # Display center coordinates of detected trees
            st.write("### Tree Center Coordinates (in original image pixels)")
            if detected_trees_list:
                tree_data = [
                    {
                        "Tree ID": i + 1,
                        "X_Center": f"{tree['center_coords_pixel'][0]:.2f}",
                        "Y_Center": f"{tree['center_coords_pixel'][1]:.2f}",
                        "Confidence": f"{tree['confidence']:.2f}",
                        "Class": tree['class']
                    }
                    for i, tree in enumerate(detected_trees_list)
                ]
                st.dataframe(tree_data, height=300) # Display in a scrollable table
            else:
                st.warning("No trees found in the selected ROI.")

    except Exception as e:
        st.error(f"An error occurred during image processing: {e}")
        st.warning("Please ensure the TIFF/PNG file is valid and not corrupted, or it might be an issue with image reading/conversion.")
        st.exception(e) # Show full traceback for detailed debugging
    finally:
        # Clean up the temporary file
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
else:
    st.info("Please upload an Orthophoto (.tiff, .png) file to begin.")

