#"""
#Preprocessing utilities for remote sensing data
#Includes image processing, band operations, and data augmentation
#"""

import numpy as np
import cv2
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import warnings
warnings.filterwarnings('ignore')

def preprocess_image(image, bands=None, normalization='min-max', 
                    denoise=False, enhance_contrast=False, target_size=None):
    """
    Comprehensive preprocessing pipeline for remote sensing images
    
    Args:
        image: Input image (numpy array)
        bands: List of band names to select
        normalization: Normalization method ('min-max', 'z-score', 'robust', 'none')
        denoise: Apply denoising
        enhance_contrast: Apply contrast enhancement
        target_size: Resize to target size (height, width)
    
    Returns:
        Preprocessed image
    """
    processed = image.copy()
    
    # Handle different input formats
    if len(processed.shape) == 2:
        processed = np.expand_dims(processed, axis=-1)
    
    # Band selection
    if bands is not None:
        processed = select_bands(processed, bands)
    
    # Resize if needed
    if target_size is not None:
        processed = resize_image(processed, target_size)
    
    # Denoising
    if denoise:
        processed = apply_denoising(processed)
    
    # Contrast enhancement
    if enhance_contrast:
        processed = enhance_image_contrast(processed)
    
    # Normalization
    processed = normalize_image(processed, method=normalization)
    
    return processed

def select_bands(image, bands):
    """
    Select specific bands from multi-spectral image
    
    Args:
        image: Input multi-spectral image
        bands: List of band names or indices
    
    Returns:
        Image with selected bands
    """
    band_mapping = {
        'Red': 0, 'R': 0,
        'Green': 1, 'G': 1,
        'Blue': 2, 'B': 2,
        'NIR': 3, 'Near-Infrared': 3,
        'SWIR1': 4, 'SWIR-1': 4,
        'SWIR2': 5, 'SWIR-2': 5,
        'Thermal': 6, 'TIR': 6
    }
    
    if isinstance(bands[0], str):
        # Convert band names to indices
        band_indices = [band_mapping.get(band, 0) for band in bands]
    else:
        # Use band indices directly
        band_indices = bands
    
    # Ensure indices are within image dimensions
    band_indices = [idx for idx in band_indices if idx < image.shape[-1]]
    
    if not band_indices:
        return image
    
    return image[:, :, band_indices]

def normalize_image(image, method='min-max'):
    """
    Normalize image using various methods
    
    Args:
        image: Input image
        method: Normalization method
    
    Returns:
        Normalized image
    """
    if method.lower() == 'none':
        return image
    
    # Convert to float32
    normalized = image.astype(np.float32)
    
    if method.lower() == 'min-max':
        # Min-Max normalization to [0, 1]
        for i in range(normalized.shape[-1]):
            band = normalized[:, :, i]
            min_val = np.min(band)
            max_val = np.max(band)
            if max_val > min_val:
                normalized[:, :, i] = (band - min_val) / (max_val - min_val)
    
    elif method.lower() == 'z-score':
        # Z-score normalization
        for i in range(normalized.shape[-1]):
            band = normalized[:, :, i]
            mean_val = np.mean(band)
            std_val = np.std(band)
            if std_val > 0:
                normalized[:, :, i] = (band - mean_val) / std_val
    
    elif method.lower() == 'robust':
        # Robust normalization using percentiles
        for i in range(normalized.shape[-1]):
            band = normalized[:, :, i]
            p25 = np.percentile(band, 25)
            p75 = np.percentile(band, 75)
            if p75 > p25:
                normalized[:, :, i] = (band - p25) / (p75 - p25)
    
    elif method.lower() == 'imagenet':
        # ImageNet normalization
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        
        # Ensure we have at least 3 channels
        if normalized.shape[-1] >= 3:
            for i in range(min(3, normalized.shape[-1])):
                normalized[:, :, i] = (normalized[:, :, i] - imagenet_mean[i]) / imagenet_std[i]
    
    return normalized

def resize_image(image, target_size, interpolation=cv2.INTER_LINEAR):
    """
    Resize image to target size
    
    Args:
        image: Input image
        target_size: Target size (height, width)
        interpolation: Interpolation method
    
    Returns:
        Resized image
    """
    if len(image.shape) == 2:
        resized = cv2.resize(image, (target_size[1], target_size[0]), 
                           interpolation=interpolation)
        return np.expand_dims(resized, axis=-1)
    else:
        return cv2.resize(image, (target_size[1], target_size[0]), 
                         interpolation=interpolation)

def apply_denoising(image, method='bilateral'):
    """
    Apply denoising to image
    
    Args:
        image: Input image
        method: Denoising method ('bilateral', 'gaussian', 'median')
    
    Returns:
        Denoised image
    """
    denoised = image.copy()
    
    if method.lower() == 'bilateral':
        # Bilateral filter preserves edges
        for i in range(denoised.shape[-1]):
            denoised[:, :, i] = cv2.bilateralFilter(
                denoised[:, :, i].astype(np.uint8), 9, 75, 75
            )
    
    elif method.lower() == 'gaussian':
        # Gaussian blur
        denoised = cv2.GaussianBlur(denoised, (5, 5), 0)
    
    elif method.lower() == 'median':
        # Median filter
        for i in range(denoised.shape[-1]):
            denoised[:, :, i] = cv2.medianBlur(
                denoised[:, :, i].astype(np.uint8), 5
            )
    
    return denoised

def enhance_image_contrast(image, method='clahe'):
    """
    Enhance image contrast
    
    Args:
        image: Input image
        method: Enhancement method ('clahe', 'histogram_eq', 'gamma')
    
    Returns:
        Contrast enhanced image
    """
    enhanced = image.copy()
    
    if method.lower() == 'clahe':
        # Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        for i in range(enhanced.shape[-1]):
            band = enhanced[:, :, i]
            # Convert to uint8 for CLAHE
            band_uint8 = ((band - band.min()) / (band.max() - band.min()) * 255).astype(np.uint8)
            enhanced[:, :, i] = clahe.apply(band_uint8) / 255.0
    
    elif method.lower() == 'histogram_eq':
        # Global histogram equalization
        for i in range(enhanced.shape[-1]):
            band = enhanced[:, :, i]
            band_uint8 = ((band - band.min()) / (band.max() - band.min()) * 255).astype(np.uint8)
            enhanced[:, :, i] = cv2.equalizeHist(band_uint8) / 255.0
    
    elif method.lower() == 'gamma':
        # Gamma correction
        gamma = 1.2
        enhanced = np.power(enhanced, gamma)
    
    return enhanced

def extract_roi(image, coordinates, buffer=0):
    """
    Extract Region of Interest from image
    
    Args:
        image: Input image
        coordinates: ROI coordinates (x, y, width, height) or (x1, y1, x2, y2)
        buffer: Buffer around ROI in pixels
    
    Returns:
        Extracted ROI
    """
    if len(coordinates) == 4:
        if coordinates[2] < coordinates[0]:  # (x1, y1, x2, y2) format
            x1, y1, x2, y2 = coordinates
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
        else:  # (x, y, width, height) format
            x, y, w, h = coordinates
    else:
        raise ValueError("Coordinates must be (x, y, width, height) or (x1, y1, x2, y2)")
    
    # Apply buffer
    x = max(0, x - buffer)
    y = max(0, y - buffer)
    w = min(image.shape[1] - x, w + 2 * buffer)
    h = min(image.shape[0] - y, h + 2 * buffer)
    
    # Extract ROI
    roi = image[y:y+h, x:x+w]
    
    return roi

def calculate_spectral_indices(image):
    """
    Calculate common spectral indices for remote sensing
    
    Args:
        image: Multi-spectral image with bands [R, G, B, NIR, SWIR1, SWIR2]
    
    Returns:
        Dictionary of spectral indices
    """
    indices = {}
    
    # Ensure we have enough bands
    if image.shape[-1] < 4:
        print("Warning: Need at least 4 bands (R, G, B, NIR) for spectral indices")
        return indices
    
    # Extract bands
    R = image[:, :, 0].astype(np.float32)
    G = image[:, :, 1].astype(np.float32)
    B = image[:, :, 2].astype(np.float32)
    NIR = image[:, :, 3].astype(np.float32)
    
    # NDVI (Normalized Difference Vegetation Index)
    indices['NDVI'] = np.divide(NIR - R, NIR + R + 1e-8, 
                               out=np.zeros_like(NIR), where=(NIR + R) != 0)
    
    # NDWI (Normalized Difference Water Index)
    indices['NDWI'] = np.divide(G - NIR, G + NIR + 1e-8,
                               out=np.zeros_like(G), where=(G + NIR) != 0)
    
    # EVI (Enhanced Vegetation Index)
    indices['EVI'] = 2.5 * np.divide(NIR - R, NIR + 6 * R - 7.5 * B + 1 + 1e-8,
                                    out=np.zeros_like(NIR), 
                                    where=(NIR + 6 * R - 7.5 * B + 1) != 0)
    
    # SAVI (Soil Adjusted Vegetation Index)
    L = 0.5  # Soil brightness correction factor
    indices['SAVI'] = np.divide((NIR - R) * (1 + L), NIR + R + L + 1e-8,
                               out=np.zeros_like(NIR), where=(NIR + R + L) != 0)
    
    if image.shape[-1] >= 5:  # If SWIR1 is available
        SWIR1 = image[:, :, 4].astype(np.float32)
        
        # NBR (Normalized Burn Ratio)
        indices['NBR'] = np.divide(NIR - SWIR1, NIR + SWIR1 + 1e-8,
                                  out=np.zeros_like(NIR), where=(NIR + SWIR1) != 0)
        
        # NDMI (Normalized Difference Moisture Index)
        indices['NDMI'] = np.divide(NIR - SWIR1, NIR + SWIR1 + 1e-8,
                                   out=np.zeros_like(NIR), where=(NIR + SWIR1) != 0)
    
    if image.shape[-1] >= 6:  # If SWIR2 is available
        SWIR2 = image[:, :, 5].astype(np.float32)
        
        # NBR2 (Normalized Burn Ratio 2)
        indices['NBR2'] = np.divide(SWIR1 - SWIR2, SWIR1 + SWIR2 + 1e-8,
                                   out=np.zeros_like(SWIR1), where=(SWIR1 + SWIR2) != 0)
    
    return indices

def apply_cloud_shadow_mask(image, cloud_mask=None, shadow_mask=None):
    """
    Apply cloud and shadow masks to image
    
    Args:
        image: Input image
        cloud_mask: Binary cloud mask
        shadow_mask: Binary shadow mask
    
    Returns:
        Masked image
    """
    masked_image = image.copy()
    
    if cloud_mask is not None:
        # Set cloud pixels to NaN or interpolate
        cloud_pixels = cloud_mask > 0
        if len(masked_image.shape) == 3:
            for i in range(masked_image.shape[-1]):
                masked_image[:, :, i][cloud_pixels] = np.nan
        else:
            masked_image[cloud_pixels] = np.nan
    
    if shadow_mask is not None:
        # Set shadow pixels to NaN or interpolate
        shadow_pixels = shadow_mask > 0
        if len(masked_image.shape) == 3:
            for i in range(masked_image.shape[-1]):
                masked_image[:, :, i][shadow_pixels] = np.nan
        else:
            masked_image[shadow_pixels] = np.nan
    
    return masked_image

def atmospheric_correction(image, method='dark_object_subtraction'):
    """
    Apply atmospheric correction to satellite imagery
    
    Args:
        image: Input satellite image
        method: Correction method
    
    Returns:
        Atmospherically corrected image
    """
    corrected = image.copy().astype(np.float32)
    
    if method.lower() == 'dark_object_subtraction':
        # Simple dark object subtraction
        for i in range(corrected.shape[-1]):
            band = corrected[:, :, i]
            # Find dark object value (1st percentile)
            dark_value = np.percentile(band[band > 0], 1)
            # Subtract dark object value
            corrected[:, :, i] = np.maximum(band - dark_value, 0)
    
    elif method.lower() == 'histogram_matching':
        # Match histogram to reference (simplified)
        for i in range(corrected.shape[-1]):
            band = corrected[:, :, i]
            # Simple linear stretch
            p2, p98 = np.percentile(band[band > 0], [2, 98])
            corrected[:, :, i] = np.clip((band - p2) / (p98 - p2), 0, 1)
    
    return corrected

def create_composite_image(images, method='median'):
    """
    Create composite image from multiple images
    
    Args:
        images: List of images
        method: Compositing method ('median', 'mean', 'max', 'min')
    
    Returns:
        Composite image
    """
    if not images:
        return None
    
    # Stack images
    image_stack = np.stack(images, axis=0)
    
    if method.lower() == 'median':
        composite = np.median(image_stack, axis=0)
    elif method.lower() == 'mean':
        composite = np.mean(image_stack, axis=0)
    elif method.lower() == 'max':
        composite = np.max(image_stack, axis=0)
    elif method.lower() == 'min':
        composite = np.min(image_stack, axis=0)
    else:
        composite = np.median(image_stack, axis=0)  # Default to median
    
    return composite

def temporal_filtering(image_series, method='median', window_size=3):
    """
    Apply temporal filtering to image time series
    
    Args:
        image_series: Time series of images
        method: Filtering method
        window_size: Temporal window size
    
    Returns:
        Temporally filtered images
    """
    filtered_series = []
    
    for i in range(len(image_series)):
        # Define temporal window
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(image_series), i + window_size // 2 + 1)
        
        window_images = image_series[start_idx:end_idx]
        
        if method.lower() == 'median':
            filtered_img = create_composite_image(window_images, 'median')
        elif method.lower() == 'mean':
            filtered_img = create_composite_image(window_images, 'mean')
        else:
            filtered_img = image_series[i]  # No filtering
        
        filtered_series.append(filtered_img)
    
    return filtered_series

def data_augmentation(image, augmentation_params=None):
    """
    Apply data augmentation for training
    
    Args:
        image: Input image
        augmentation_params: Dictionary of augmentation parameters
    
    Returns:
        Augmented image
    """
    if augmentation_params is None:
        augmentation_params = {
            'rotation': 15,
            'flip_horizontal': True,
            'flip_vertical': False,
            'brightness': 0.2,
            'contrast': 0.2,
            'noise': 0.02
        }
    
    augmented = image.copy()
    
    # Random rotation
    if augmentation_params.get('rotation', 0) > 0:
        angle = np.random.uniform(-augmentation_params['rotation'], 
                                  augmentation_params['rotation'])
        h, w = augmented.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        if len(augmented.shape) == 3:
            for i in range(augmented.shape[-1]):
                augmented[:, :, i] = cv2.warpAffine(
                    augmented[:, :, i], rotation_matrix, (w, h)
                )
        else:
            augmented = cv2.warpAffine(augmented, rotation_matrix, (w, h))
    
    # Horizontal flip
    if augmentation_params.get('flip_horizontal', False) and np.random.random() > 0.5:
        augmented = cv2.flip(augmented, 1)
    
    # Vertical flip
    if augmentation_params.get('flip_vertical', False) and np.random.random() > 0.5:
        augmented = cv2.flip(augmented, 0)
    
    # Brightness adjustment
    if augmentation_params.get('brightness', 0) > 0:
        brightness_factor = 1 + np.random.uniform(
            -augmentation_params['brightness'], 
            augmentation_params['brightness']
        )
        augmented = np.clip(augmented * brightness_factor, 0, 1)
    
    # Contrast adjustment
    if augmentation_params.get('contrast', 0) > 0:
        contrast_factor = 1 + np.random.uniform(
            -augmentation_params['contrast'], 
            augmentation_params['contrast']
        )
        mean_val = np.mean(augmented)
        augmented = np.clip((augmented - mean_val) * contrast_factor + mean_val, 0, 1)
    
    # Add noise
    if augmentation_params.get('noise', 0) > 0:
        noise = np.random.normal(0, augmentation_params['noise'], augmented.shape)
        augmented = np.clip(augmented + noise, 0, 1)
    
    return augmented

def quality_assessment(image, mask=None):
    """
    Assess image quality metrics
    
    Args:
        image: Input image
        mask: Quality mask (optional)
    
    Returns:
        Dictionary of quality metrics
    """
    quality_metrics = {}
    
    # Convert to grayscale for some metrics
    if len(image.shape) == 3:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (image * 255).astype(np.uint8)
    
    # Sharpness (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    quality_metrics['sharpness'] = laplacian.var()
    
    # Contrast (standard deviation)
    quality_metrics['contrast'] = np.std(gray)
    
    # Signal-to-noise ratio (simplified)
    signal = np.mean(gray)
    noise = np.std(gray)
    quality_metrics['snr'] = signal / (noise + 1e-8)
    
    # Cloud coverage (if mask provided)
    if mask is not None:
        quality_metrics['cloud_coverage'] = np.mean(mask) * 100
    
    # Valid pixel percentage
    if len(image.shape) == 3:
        valid_pixels = ~np.isnan(image).any(axis=-1)
    else:
        valid_pixels = ~np.isnan(image)
    
    quality_metrics['valid_pixel_percentage'] = np.mean(valid_pixels) * 100
    
    return quality_metrics

def geospatial_preprocessing(image_path, target_crs=None, target_resolution=None):
    """
    Geospatial preprocessing using rasterio
    
    Args:
        image_path: Path to geospatial image
        target_crs: Target coordinate reference system
        target_resolution: Target resolution in meters
    
    Returns:
        Preprocessed image and metadata
    """
    try:
        import rasterio
        from rasterio.warp import reproject, Resampling
        from rasterio.crs import CRS
        
        with rasterio.open(image_path) as src:
            # Read image
            image = src.read()
            
            # Get metadata
            metadata = {
                'crs': src.crs,
                'transform': src.transform,
                'width': src.width,
                'height': src.height,
                'count': src.count,
                'dtype': src.dtype,
                'bounds': src.bounds
            }
            
            # Reproject if target CRS is specified
            if target_crs is not None:
                target_crs = CRS.from_string(target_crs)
                if src.crs != target_crs:
                    # Calculate target transform and dimensions
                    from rasterio.warp import calculate_default_transform
                    
                    transform, width, height = calculate_default_transform(
                        src.crs, target_crs, src.width, src.height, *src.bounds
                    )
                    
                    # Create destination array
                    dest_image = np.zeros((src.count, height, width), dtype=src.dtype)
                    
                    # Reproject
                    reproject(
                        source=image,
                        destination=dest_image,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.bilinear
                    )
                    
                    image = dest_image
                    metadata.update({
                        'crs': target_crs,
                        'transform': transform,
                        'width': width,
                        'height': height
                    })
            
            # Resample if target resolution is specified
            if target_resolution is not None:
                current_res = abs(src.transform[0])  # Assuming square pixels
                scale_factor = current_res / target_resolution
                
                if scale_factor != 1.0:
                    new_width = int(src.width * scale_factor)
                    new_height = int(src.height * scale_factor)
                    
                    # Resample each band
                    resampled_image = np.zeros((src.count, new_height, new_width), 
                                             dtype=src.dtype)
                    
                    for i in range(src.count):
                        resampled_image[i] = cv2.resize(
                            image[i], (new_width, new_height),
                            interpolation=cv2.INTER_LINEAR
                        )
                    
                    image = resampled_image
                    
                    # Update transform
                    new_transform = src.transform * src.transform.scale(
                        src.width / new_width, src.height / new_height
                    )
                    
                    metadata.update({
                        'transform': new_transform,
                        'width': new_width,
                        'height': new_height
                    })
            
            # Convert to standard format (H, W, C)
            if len(image.shape) == 3:
                image = np.transpose(image, (1, 2, 0))
            
            return image, metadata
    
    except ImportError:
        print("rasterio not available. Using basic image loading.")
        import PIL.Image
        image = np.array(PIL.Image.open(image_path))
        metadata = {'source': image_path}
        return image, metadata

def batch_preprocessing(image_list, preprocessing_params):
    """
    Apply preprocessing to a batch of images
    
    Args:
        image_list: List of images or image paths
        preprocessing_params: Dictionary of preprocessing parameters
    
    Returns:
        List of preprocessed images
    """
    processed_images = []
    
    for image in image_list:
        # Load image if path is provided
        if isinstance(image, str):
            try:
                import PIL.Image
                image = np.array(PIL.Image.open(image))
            except:
                print(f"Could not load image: {image}")
                continue
        
        # Apply preprocessing
        processed = preprocess_image(image, **preprocessing_params)
        processed_images.append(processed)
    
    return processed_images


# Example usage and testing
if __name__ == "__main__":
    print("Remote Sensing Preprocessing Utilities - Example Usage")
    
    # Create synthetic multi-spectral image
    print("\n1. Creating synthetic multi-spectral image...")
    height, width = 256, 256
    num_bands = 6  # R, G, B, NIR, SWIR1, SWIR2
    
    synthetic_image = np.random.randint(0, 255, (height, width, num_bands)).astype(np.uint8)
    print(f"Synthetic image shape: {synthetic_image.shape}")
    
    # Test band selection
    print("\n2. Testing band selection...")
    rgb_bands = select_bands(synthetic_image, ['Red', 'Green', 'Blue'])
    print(f"RGB bands shape: {rgb_bands.shape}")
    
    nir_rgb_bands = select_bands(synthetic_image, ['NIR', 'Red', 'Green'])
    print(f"NIR-RGB bands shape: {nir_rgb_bands.shape}")
    
    # Test normalization methods
    print("\n3. Testing normalization methods...")
    norm_methods = ['min-max', 'z-score', 'robust']
    
    for method in norm_methods:
        normalized = normalize_image(rgb_bands, method=method)
        print(f"{method} normalization - range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    # Test spectral indices
    print("\n4. Calculating spectral indices...")
    indices = calculate_spectral_indices(synthetic_image)
    
    for index_name, index_array in indices.items():
        print(f"{index_name}: shape {index_array.shape}, range [{index_array.min():.3f}, {index_array.max():.3f}]")
    
    # Test preprocessing pipeline
    print("\n5. Testing complete preprocessing pipeline...")
    processed = preprocess_image(
        synthetic_image,
        bands=['Red', 'Green', 'Blue', 'NIR'],
        normalization='min-max',
        denoise=True,
        enhance_contrast=True,
        target_size=(224, 224)
    )
    print(f"Processed image shape: {processed.shape}")
    print(f"Processed image range: [{processed.min():.3f}, {processed.max():.3f}]")
    
    # Test ROI extraction
    print("\n6. Testing ROI extraction...")
    roi_coords = (50, 50, 100, 100)  # x, y, width, height
    roi = extract_roi(processed, roi_coords, buffer=10)
    print(f"ROI shape: {roi.shape}")
    
    # Test data augmentation
    print("\n7. Testing data augmentation...")
    augmentation_params = {
        'rotation': 15,
        'flip_horizontal': True,
        'brightness': 0.2,
        'contrast': 0.1,
        'noise': 0.02
    }
    
    augmented = data_augmentation(processed, augmentation_params)
    print(f"Augmented image shape: {augmented.shape}")
    print(f"Augmented image range: [{augmented.min():.3f}, {augmented.max():.3f}]")
    
    # Test quality assessment
    print("\n8. Testing quality assessment...")
    quality = quality_assessment(processed)
    print("Quality metrics:")
    for metric, value in quality.items():
        print(f"  {metric}: {value:.3f}")
    
    # Test batch preprocessing
    print("\n9. Testing batch preprocessing...")
    image_batch = [synthetic_image, synthetic_image, synthetic_image]
    
    batch_params = {
        'bands': ['Red', 'Green', 'Blue'],
        'normalization': 'min-max',
        'target_size': (128, 128)
    }
    
    processed_batch = batch_preprocessing(image_batch, batch_params)
    print(f"Processed batch: {len(processed_batch)} images")
    print(f"Each image shape: {processed_batch[0].shape}")
    
    print("\nPreprocessing utilities ready!")
    print("Key features:")
    print("- Multi-spectral band selection")
    print("- Various normalization methods")
    print("- Spectral indices calculation")
    print("- Denoising and contrast enhancement")
    print("- ROI extraction")
    print("- Data augmentation")
    print("- Quality assessment")
    print("- Batch processing support")