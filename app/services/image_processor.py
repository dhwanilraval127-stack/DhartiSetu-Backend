"""
Image Processing Service (Optimized)
"""
import cv2
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Optimized image preprocessing for CNN models using OpenCV"""
    
    def load_and_preprocess(
        self,
        image_bytes: bytes,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True
    ) -> np.ndarray:
        """
        Optimized image loading and preprocessing using OpenCV
        Much faster than PIL for resizing and conversion
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            # Decode image using OpenCV (much faster than PIL)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Could not decode image - invalid format")
            
            # Convert BGR to RGB (OpenCV uses BGR by default)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize using efficient interpolation
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            
            # Convert to float32
            img = img.astype(np.float32)
            
            # Normalize if required
            if normalize:
                img = img / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            return img
            
        except cv2.error as e:
            raise ValueError(f"OpenCV error processing image: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")
    
    def validate_image(self, image_bytes: bytes, max_size: int = 10 * 1024 * 1024) -> Tuple[bool, str]:
        """
        Fast image validation using header checks
        """
        try:
            # Quick size check
            if len(image_bytes) == 0:
                return False, "Empty file"
            
            if len(image_bytes) > max_size:
                return False, f"Image size ({len(image_bytes)} bytes) exceeds maximum allowed ({max_size} bytes)"
            
            # Quick header validation for common formats
            header = image_bytes[:32]  # Check first 32 bytes
            
            # JPEG validation
            if header.startswith(b'\xff\xd8\xff'):
                return True, "Valid JPEG image"
            
            # PNG validation
            if header.startswith(b'\x89PNG\r\n\x1a\n'):
                return True, "Valid PNG image"
            
            # Try to decode with OpenCV for final validation
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                return True, "Valid image"
            else:
                return False, "Unsupported or corrupted image format"
                
        except Exception as e:
            return False, f"Image validation error: {str(e)}"

# Create singleton instance
image_processor = ImageProcessor()