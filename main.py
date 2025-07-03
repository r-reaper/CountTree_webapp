import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
import tempfile
import os
import base64
import rasterio # Import rasterio for GeoTIFF handling and coordinate transformation

# Import torch and torchvision for global NMS
import torch
import torchvision.ops as ops 

from ultralytics import YOLO
from streamlit_drawable_canvas import st_canvas

# --- Function to load YOLO model ---
@st.cache_resource
def load_yolo_model(model_path='best.pt'):
    """
    Loads the YOLO model from the specified path.
    Uses st.cache_resource to cache the model, preventing reloads on every interaction.
    """
    try:
        # Check if the model file exists
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            st.error("Please ensure 'best.pt' is in the same directory as your app.py or provide the full path.")
            st.stop()

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
def detect_trees_in_roi(model, roi_image_np, roi_offset_x, roi_offset_y, transform, TILE_SIZE=640, OVERLAP=100, conf_threshold=0.25, iou_threshold=0.7):
    """
    Detects trees within a specified Region of Interest (ROI) by tiling the ROI
    and running YOLO inference on each tile.
    Applies global NMS and converts pixel coordinates to geographic coordinates.
    """
    all_detections_original_coords = [] # Stores [x_min, y_min, x_max, y_max, confidence, class_id] in original image pixel coords
    roi_height, roi_width, _ = roi_image_np.shape

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
            # NMS is applied per tile here, but we need global NMS later
            results = model(padded_tile, imgsz=TILE_SIZE, conf=conf_threshold) # Only conf here, IoU for global NMS

            # Process detection results for current tile
            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                confidences = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()

                for i in range(len(boxes)):
                    x_min_rel, y_min_rel, x_max_rel, y_max_rel = boxes[i]
                    conf = confidences[i]
                    cls_id = int(classes[i])

                    # Convert tile-relative coordinates to original image pixel coordinates
                    x_min_orig = roi_offset_x + x_start_tile + x_min_rel
                    y_min_orig = roi_offset_y + y_start_tile + y_min_rel
                    x_max_orig = roi_offset_x + x_start_tile + x_max_rel
                    y_max_orig = roi_offset_y + y_start_tile + y_max_rel

                    all_detections_original_coords.append([x_min_orig, y_min_orig, x_max_orig, y_max_orig, conf, cls_id])

    # --- Apply Global Non-Maximum Suppression (NMS) ---
    if not all_detections_original_coords:
        # If no detections, return empty list and original ROI image
        return [], roi_image_np.copy()

    all_detections_np = np.array(all_detections_original_coords)
    
    # Extract boxes, scores, and class IDs
    boxes_np = all_detections_np[:, :4]
    scores_np = all_detections_np[:, 4]
    classes_np = all_detections_np[:, 5]

    # Convert to PyTorch tensors
    boxes_tensor = torch.from_numpy(boxes_np).float()
    scores_tensor = torch.from_numpy(scores_np).float()
    classes_tensor = torch.from_numpy(classes_np).long()

    # Perform NMS per class to ensure different classes don't suppress each other
    keep_indices = []
    unique_classes = torch.unique(classes_tensor)
    for cls_id in unique_classes:
        # Get indices for the current class
        class_specific_indices = (classes_tensor == cls_id).nonzero(as_tuple=True)[0]
        
        if len(class_specific_indices) > 0:
            # Get boxes and scores for the current class
            class_boxes = boxes_tensor[class_specific_indices]
            class_scores = scores_tensor[class_specific_indices]
            
            # Apply NMS using the global IoU threshold
            nms_retained_indices_in_class = ops.nms(class_boxes, class_scores, iou_threshold)
            
            # Map back to original indices
            keep_indices.extend(class_specific_indices[nms_retained_indices_in_class].tolist())

    # Filter the detections using the indices kept by NMS
    final_filtered_detections = all_detections_np[keep_indices]

    detected_trees = []
    # Create a fresh copy of the ROI image for drawing only the NMS-filtered boxes
    display_roi_image_with_bboxes = roi_image_np.copy()

    # --- Process and draw filtered detections ---
    for det in final_filtered_detections:
        x_min_orig, y_min_orig, x_max_orig, y_max_orig, conf, cls_id = det
        class_name = model.names[int(cls_id)]

        center_x_pixel = (x_min_orig + x_max_orig) / 2
        center_y_pixel = (y_min_orig + y_max_orig) / 2

        # Convert pixel coordinates to geographic coordinates using the transform
        center_lon, center_lat = None, None
        if transform is not None:
            center_lon, center_lat = rasterio.transform.xy(transform, center_y_pixel, center_x_pixel)

        detected_trees.append({
            'bbox_pixel': [int(x_min_orig), int(y_min_orig), int(x_max_orig), int(y_max_orig)],
            'confidence': conf,
            'class': class_name,
            'center_coords_pixel': (center_x_pixel, center_y_pixel),
            'center_coords_geo': (center_lon, center_lat) # Add geographic coordinates
        })
        
        # Draw bounding box and center on the display_roi_image_with_bboxes
        # Convert original image pixel coordinates back to ROI-relative coordinates for drawing
        x_min_roi_draw = int(x_min_orig - roi_offset_x)
        y_min_roi_draw = int(y_min_orig - roi_offset_y)
        x_max_roi_draw = int(x_max_orig - roi_offset_x)
        y_max_roi_draw = int(y_max_orig - roi_offset_y)
        center_x_roi_draw = int(center_x_pixel - roi_offset_x)
        center_y_roi_draw = int(center_y_pixel - roi_offset_y)

        # Ensure drawing coordinates are within bounds of display_roi_image_with_bboxes
        x_min_roi_draw = max(0, x_min_roi_draw)
        y_min_roi_draw = max(0, y_min_roi_draw)
        x_max_roi_draw = min(roi_width, x_max_roi_draw)
        y_max_roi_draw = min(roi_height, y_max_roi_draw)

        # Check if coordinates are valid for rectangle drawing
        if x_min_roi_draw < x_max_roi_draw and y_min_roi_draw < y_max_roi_draw:
            cv2.rectangle(display_roi_image_with_bboxes, (x_min_roi_draw, y_min_roi_draw), (x_max_roi_draw, y_max_roi_draw), (0, 255, 0), 2) # Green box
            cv2.circle(display_roi_image_with_bboxes, (center_x_roi_draw, center_y_roi_draw), 3, (0, 0, 255), -1) # Red dot

    return detected_trees, display_roi_image_with_bboxes


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
        value=0.5, # Default value
        step=0.05,
        help="Minimum confidence score for a detection to be considered valid (0.0 - 1.0)"
    )
    # Slider for IoU Threshold (for Non-Maximum Suppression)
    iou_threshold = st.slider(
        "Set IoU Threshold (Global NMS)",
        min_value=0.0,
        max_value=1.0,
        value=0.4, # Default value, often lower for NMS
        step=0.05,
        help="Maximum Intersection Over Union (IoU) for global non-maximum suppression (NMS) to filter overlapping bounding boxes. A lower value means stricter filtering (fewer overlaps)."
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

        # Open image using rasterio to get georeferencing information
        # Use rasterio.open for TIFF files to get transform and CRS
        if uploaded_file.type in ["image/tiff", "image/tif"]:
            with rasterio.open(temp_image_path) as src:
                # Read specific bands if it's a multi-band image, assuming RGB order for display/YOLO
                if src.count >= 3:
                    # Try to read RGB bands, assuming common order or first 3 bands
                    try:
                        # Attempt to read bands 1, 2, 3 as R, G, B
                        original_image_np = src.read([1, 2, 3]).transpose((1, 2, 0))
                    except Exception:
                        # Fallback to reading all bands if specific band reading fails or less than 3 bands
                        original_image_np = src.read().transpose((1, 2, 0))
                else: # Grayscale or less than 3 bands
                    original_image_np = src.read().transpose((1, 2, 0))

                image_transform = src.transform # Geotransform for coordinate transformation
                image_crs = src.crs # Coordinate Reference System
            st.info(f"Loaded TIFF with CRS: {image_crs.to_string()}")
        else: # For PNG or other image types, assume no georeferencing
            original_image_pil = Image.open(temp_image_path)
            original_image_np = np.array(original_image_pil)
            image_transform = None # No transform for non-GeoTIFF
            image_crs = None
            st.warning("Non-TIFF file uploaded. Geographic coordinates will not be available.")
        
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
        
        # Pass PIL Image directly to background_image
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
            # Check if transform is available for geographic coordinates
            if uploaded_file.type in ["image/tiff", "image/tif"] and image_transform is None:
                st.error("Cannot calculate real-world coordinates: Georeferencing information not found or could not be read from the uploaded TIFF file.")
                st.warning("Please ensure the TIFF file is a valid GeoTIFF.")
            
            with st.spinner("Detecting trees... Please wait."):
                # Call the detection function with selected thresholds and image_transform
                detected_trees_list, roi_display_image_with_bboxes = detect_trees_in_roi(
                    model,
                    roi_image,
                    roi_offset_x=roi_x_min,
                    roi_offset_y=roi_y_min,
                    transform=image_transform, # Pass the georeferencing transform
                    conf_threshold=confidence_threshold, # Pass confidence from slider
                    iou_threshold=iou_threshold # Pass IoU from slider (now used for global NMS)
                )

            st.subheader("Tree Detection Results")
            st.write(f"Total trees detected: **{len(detected_trees_list)}** trees")

            # Display ROI image with bounding boxes
            st.image(roi_display_image_with_bboxes, caption="ROI Image with Detections", use_column_width=True)

            # Display center coordinates of detected trees
            st.write("### Tree Center Coordinates")
            if detected_trees_list:
                tree_data = []
                for i, tree in enumerate(detected_trees_list):
                    row_data = {
                        "Tree ID": i + 1,
                        "X_Pixel": f"{tree['center_coords_pixel'][0]:.2f}",
                        "Y_Pixel": f"{tree['center_coords_pixel'][1]:.2f}",
                        "Confidence": f"{tree['confidence']:.2f}",
                        "Class": tree['class']
                    }
                    if 'center_coords_geo' in tree and tree['center_coords_geo'][0] is not None and tree['center_coords_geo'][1] is not None:
                        row_data["Longitude"] = f"{tree['center_coords_geo'][0]:.6f}"
                        row_data["Latitude"] = f"{tree['center_coords_geo'][1]:.6f}"
                    else:
                        row_data["Longitude"] = "N/A"
                        row_data["Latitude"] = "N/A"
                    tree_data.append(row_data)
                
                st.dataframe(tree_data, height=300) # Display in a scrollable table
                
                if image_crs:
                    st.info(f"Geographic coordinates are in CRS: {image_crs.to_string()}")
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

