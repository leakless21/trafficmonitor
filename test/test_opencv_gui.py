import cv2
import numpy as np
from loguru import logger


def test_opencv_gui():
    """
    Test OpenCV GUI functionality including window creation, image display,
    drawing operations, and keyboard interaction.
    """
    logger.info("Starting OpenCV GUI test...")
    
    try:
        # Create a test image (640x480, 3-channel BGR)
        height, width = 480, 640
        test_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Fill with a gradient background
        for y in range(height):
            test_image[y, :] = [int(255 * y / height), 100, int(255 * (1 - y / height))]
        
        # Draw some shapes and text
        cv2.rectangle(test_image, (50, 50), (200, 150), (0, 255, 0), 2)
        cv2.circle(test_image, (320, 240), 80, (255, 0, 0), -1)
        cv2.ellipse(test_image, (500, 100), (80, 40), 45, 0, 360, (0, 255, 255), 3)
        cv2.line(test_image, (0, 0), (width, height), (255, 255, 255), 2)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(test_image, 'OpenCV GUI Test', (50, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(test_image, 'Press ESC to exit', (50, height - 20), font, 0.7, (255, 255, 255), 2)
        cv2.putText(test_image, 'Press SPACE for new colors', (50, height - 50), font, 0.7, (255, 255, 255), 2)
        
        # Create window
        window_name = 'OpenCV GUI Test Window'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        logger.info("OpenCV window created successfully")
        logger.info("Controls: ESC to exit, SPACE for new random colors")
        
        # Main loop
        while True:
            cv2.imshow(window_name, test_image)
            key = cv2.waitKey(30) & 0xFF
            
            if key == 27:  # ESC key
                logger.info("ESC pressed, exiting...")
                break
            elif key == 32:  # SPACE key
                logger.info("SPACE pressed, generating new colors...")
                # Generate random colors for shapes
                rect_color = tuple(int(x) for x in np.random.randint(0, 256, 3))
                circle_color = tuple(int(x) for x in np.random.randint(0, 256, 3))
                ellipse_color = tuple(int(x) for x in np.random.randint(0, 256, 3))
                
                # Redraw with new colors
                test_image = np.zeros((height, width, 3), dtype=np.uint8)
                for y in range(height):
                    test_image[y, :] = [int(255 * y / height), 100, int(255 * (1 - y / height))]
                
                cv2.rectangle(test_image, (50, 50), (200, 150), rect_color, 2)
                cv2.circle(test_image, (320, 240), 80, circle_color, -1)
                cv2.ellipse(test_image, (500, 100), (80, 40), 45, 0, 360, ellipse_color, 3)
                cv2.line(test_image, (0, 0), (width, height), (255, 255, 255), 2)
                
                cv2.putText(test_image, 'OpenCV GUI Test', (50, 30), font, 1, (255, 255, 255), 2)
                cv2.putText(test_image, 'Press ESC to exit', (50, height - 20), font, 0.7, (255, 255, 255), 2)
                cv2.putText(test_image, 'Press SPACE for new colors', (50, height - 50), font, 0.7, (255, 255, 255), 2)
        
        cv2.destroyAllWindows()
        logger.info("OpenCV GUI test completed successfully")
        return True
        
    except Exception as e:
        logger.exception(f"OpenCV GUI test failed: {e}")
        return False


def test_opencv_image_loading():
    """
    Test OpenCV image loading capabilities with a generated test image.
    """
    logger.info("Testing OpenCV image loading...")
    
    try:
        # Create and save a test image
        test_img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
        test_path = "test_opencv_image.jpg"
        
        # Save image
        cv2.imwrite(test_path, test_img)
        logger.info(f"Test image saved to {test_path}")
        
        # Load image back
        loaded_img = cv2.imread(test_path)
        if loaded_img is not None:
            logger.info(f"Image loaded successfully: shape {loaded_img.shape}")
            
            # Display loaded image
            cv2.imshow('Loaded Test Image', loaded_img)
            logger.info("Press any key to close the loaded image window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Clean up
            import os
            os.remove(test_path)
            logger.info("Test image file cleaned up")
            
            return True
        else:
            logger.error("Failed to load test image")
            return False
            
    except Exception as e:
        logger.exception(f"Image loading test failed: {e}")
        return False


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("OpenCV GUI Test Suite")
    logger.info("=" * 50)
    
    # Test basic GUI functionality
    gui_success = test_opencv_gui()
    
    # Test image loading
    loading_success = test_opencv_image_loading()
    
    # Summary
    logger.info("=" * 50)
    logger.info("Test Results:")
    logger.info(f"GUI Test: {'PASSED' if gui_success else 'FAILED'}")
    logger.info(f"Image Loading Test: {'PASSED' if loading_success else 'FAILED'}")
    
    if gui_success and loading_success:
        logger.info("All OpenCV GUI tests PASSED!")
    else:
        logger.warning("Some OpenCV GUI tests FAILED!")
    
    logger.info("=" * 50) 