from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import cv2
import numpy as np
from matplotlib import pyplot as plt
import logging
import os
from tkinter.colorchooser import askcolor

def get_file_path():
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = askopenfilename(title="Select a file")
    return file_path

def save_file(image):
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
    if file_path:
        cv2.imwrite(file_path, image)
        print(f"Image saved to {file_path}")
    else:
        print("Save operation cancelled.")

def get_user_drawn_rect(image):
    """
    Opens an OpenCV window allowing the user to draw a bounding box.

    Args:
        image (np.ndarray): The input image

    Returns:
        tuple: (x, y, w, h) bounding box
    """
    clone = image.copy()
    rect = []
    drawing = False
    ix, iy = -1, -1

    def draw_rectangle(event, x, y, flags, param):
        nonlocal ix, iy, drawing, rect, clone

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            temp = clone.copy()
            cv2.rectangle(temp, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Draw bounding box (Press ENTER to confirm)", temp)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            rect = [min(ix, x), min(iy, y), abs(x - ix), abs(y - iy)]
            cv2.rectangle(clone, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 255, 0), 2)
            cv2.imshow("Draw bounding box (Press ENTER to confirm)", clone)

    # Open window and set callback
    cv2.namedWindow("Draw bounding box (Press ENTER to confirm)")
    cv2.setMouseCallback("Draw bounding box (Press ENTER to confirm)", draw_rectangle)
    cv2.imshow("Draw bounding box (Press ENTER to confirm)", image)

    print("Instructions:")
    print("1. Click and drag to draw a bounding box.")
    print("2. Press ENTER or SPACE to confirm.")
    print("3. Press ESC to cancel.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == 32:  # ENTER or SPACE
            break
        elif key == 27:  # ESC
            rect = []
            break

    cv2.destroyAllWindows()

    if len(rect) == 4:
        logging.info(f"User-drawn rectangle: {rect}")
        return tuple(rect)
    else:
        logging.warning("Bounding box selection cancelled.")

def apply_grabcut(image, rect=None, iter_count=5):
    """
    Applies the GrabCut algorithm to extract the foreground.

    Args:
        image (np.ndarray): Input image (BGR)
        rect (tuple): Bounding box in the format (x, y, w, h)
        iter_count (int): Number of GrabCut iterations

    Returns:
        tuple: (mask, foreground result)
    """
    try:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)  # 0=bg, 1=fg, 2=prob.bg, 3=prob.g
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        if rect is None:
            raise ValueError("Bounding box (rect) is required for GrabCut.")

        # Apply GrabCut with rectangle
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, iterCount=iter_count, mode=cv2.GC_INIT_WITH_RECT)

        # Convert mask to binary: 0 and 2 are background, 1 and 3 are foreground
        output_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        foreground = image * output_mask[:, :, np.newaxis]

        return output_mask * 255, foreground, mask

    except Exception as e:
        logging.error(f"GrabCut failed: {e}")
        return None, None

def refine_mask(mask, kernel_size=7, blur_size=7, iterations=7):
    """
    Cleans and smooths a binary mask.

    Args:
        mask (np.ndarray): Binary mask (0 or 255)
        kernel_size (int): Size of morphological kernel
        blur_size (int): Size of Gaussian blur kernel
        iterations (int): Dilation iterations

    Returns:
        np.ndarray: Refined mask
    """
    try:
        # Convert to 0/1 mask if needed
        binary_mask = (mask > 0).astype(np.uint8)

        # Morph kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Fill small holes and remove noise
        closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

        # Optional dilation to recover lost details (e.g. fingers, wires)
        dilated = cv2.dilate(opened, kernel, iterations=iterations)

        # Feather the edges
        blurred = cv2.GaussianBlur(dilated.astype(np.float32), (blur_size, blur_size), 0)
        mask2 = np.zeros_like(blurred)

        # Scale to [0, 255] and return
        refined = (blurred * 255).astype(np.uint8)
        return refined

    except Exception as e:
        logging.error(f"Mask refinement failed: {e}")
        return mask
    
def replace_with_solid_color(image, mask, color=(255, 255, 255)):
    """
    Replaces the background of the image with a solid BGR color.

    Args:
        image (np.ndarray): Input image
        mask (np.ndarray): Refined mask (0-255)
        color (tuple): BGR color tuple (e.g., white=(255,255,255))

    Returns:
        np.ndarray: Image with solid background
    """
    try:
        background = np.full_like(image, color, dtype=np.uint8)
        mask_3ch = cv2.merge([mask // 255] * 3)  # Convert to 3-channel binary mask
        result = (image * mask_3ch) + (background * (1 - mask_3ch))
        return result
    except Exception as e:
        logging.error(f"Solid color replacement failed: {e}")
        return None

def get_user_manual_mask(image):
    """
    Opens an OpenCV window allowing the user to draw a bounding box.

    Args:
        image (np.ndarray): The input image

    Returns:
        np.ndarray: Binary mask
    """
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clone = image.copy()
        
    drawing = False
    ix, iy = -1, -1
    val = 255
    mask = np.ones_like(gray, dtype=np.uint8)
    has_drawn = False
    print("Instructions:")
    print("1. Click and drag to draw a bounding box.")
    print("2. Press ENTER or SPACE to confirm.")
    print("3. Press ESC to cancel.")

    def draw_manual_mask(event, x, y, flags, param):
        nonlocal ix, iy, drawing, has_drawn, mask, clone, val
    
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            has_drawn = True
            val = 0
            ix, iy = x, y
        elif event == cv2.EVENT_RBUTTONDOWN:
            drawing = True
            has_drawn = True
            val = 255
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            temp = clone.copy()            
            cv2.line(mask, (ix, iy), (x, y), val, 2)
            cv2.line(clone, (ix, iy), (x, y), (0, val, 0), 2)
            cv2.imshow("Draw Manual mask (Press ENTER to confirm)", temp)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False 
        
        elif event == cv2.EVENT_RBUTTONUP:
            drawing = False            

    # Open window and set callback
    cv2.namedWindow("Draw Manual mask (Press ENTER to confirm)")
    cv2.setMouseCallback("Draw Manual mask (Press ENTER to confirm)", draw_manual_mask)
    cv2.imshow("Draw Manual mask (Press ENTER to confirm)", image) 

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == 32:  # ENTER or SPACE
            break
        elif key == 27:  # ESC
            mask = []
            break
            
    cv2.destroyAllWindows()

    return mask, has_drawn

def apply_transparency(image, mask):
    """
    Applies mask to image and returns a 4-channel BGRA image (transparent background).

    Args:
        image (np.ndarray): Input BGR image
        mask (np.ndarray): Refined mask, values in [0, 255]

    Returns:
        np.ndarray: Image with alpha channel (BGRA)
    """
    try:
        h, w, channels = image.shape
        if channels < 4:
            transparent = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        else:
            transparent = image.copy()        
        transparent[:,:, 3] = mask
        return transparent
    except Exception as e:
        logging.error(f"Failed to apply transparency: {e}")
        return None
def replace_background_with_image(image, mask, background_image):
    """
    Replaces background of the subject with a new image.

    Args:
        image (np.ndarray): Original image (BGR)
        mask (np.ndarray): Refined mask (0-255)
        background_image (np.ndarray): New background (must match dimensions)

    Returns:
        np.ndarray: Composite image
    """
    try:
        # Resize background to match input
        background_resized = cv2.resize(background_image, (image.shape[1], image.shape[0]))
        mask_3ch = cv2.merge([mask // 255] * 3)

        # Composite
        result = (image * mask_3ch) + (background_resized * (1 - mask_3ch))
        return result
    except Exception as e:
        logging.error(f"Background replacement failed: {e}")
        return None
    
def main():
    file_path = get_file_path()
    if file_path:
        print(f"Selected file: {file_path}")
    else:
        print("No file selected.")
    image = cv2.imread(file_path)
    if image is None:
        print("Error loading image.")
        return
    user_drawn_rectangle = get_user_drawn_rect(image)

    if user_drawn_rectangle:
        mask, foreground, mask_init = apply_grabcut(image, user_drawn_rectangle)
        if mask is not None:
            refined_mask = refine_mask(mask)
            final_result = cv2.bitwise_and(image, image, mask=(refined_mask // 255))

            # Save the refined mask
            while True:
                user_drawn_mask, has_drawn = get_user_manual_mask(final_result)
                if not has_drawn:
                    print("No mask drawn. Exiting.")
                    break
                else:
                    mask_init[user_drawn_mask == 0] = 0
                    mask_init[user_drawn_mask == 255] = 1
                    bgdModel = np.zeros((1,65), dtype=np.float64)
                    fgdModel = np.zeros((1,65), dtype=np.float64)
                    mask, bgdModel, fgdModel = cv2.grabCut(image,mask_init,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
                    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
                    mask2 = mask2 * 255
                    refined_mask = refine_mask(mask2, kernel_size=5, blur_size=7, iterations=3)
                        # Apply refined mask
                    final_result = cv2.bitwise_and(image, image, mask=(refined_mask // 255))
            print("Select an option:")
            print("1. Transparent background.")
            print("2. Color background.")
            print("3. Image background.")
            option = -1
            while option not in [1, 2, 3]:
                try:
                    option = int(input("Enter your choice (1/2/3): "))
                except ValueError:
                    print("Invalid input. Please enter 1, 2, or 3.")
            if option == 1:
                final_result = apply_transparency(image, refined_mask)
            elif option == 2:
                selected_color = askcolor(title="Choose a color")[0]
                print(f"Selected color: {selected_color}")
                final_result = replace_with_solid_color(image, refined_mask, color=selected_color)  # Example: Red background
            elif option == 3:
                bg_image = get_file_path()
                if bg_image:
                    background_image = cv2.imread(bg_image)
                    if background_image is None:
                        print("Error loading background image.")
                        return
                    final_result = replace_background_with_image(image, refined_mask, background_image)
                else:
                    print("No background image selected.")
                    return
            save_file(final_result)
            print("Image saved successfully.")
            cv2.imshow("Final Result", final_result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("GrabCut failed.")
    else:
        print("Bounding box not selected.")

    

if __name__ == "__main__":
    main()