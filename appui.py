import tkinter as tk
from tkinter import filedialog, colorchooser, messagebox
from PIL import Image, ImageTk, ImageOps
import numpy as np
import cv2
from imageprocessor import ImageProcessor
from colorbackgroundprocessor import ColorBackgroundProcessor
from transparentprocessor import TransparentProcessor

class AppUI:
    """Main UI class for the application."""
        

    def __init__(self, root:tk.Tk):
        self.root = root
        self.img_processor = ImageProcessor()
        self.colorbg_processor = ColorBackgroundProcessor()
        self.transparent_processor = TransparentProcessor()
        self.background_image = None
        self.bounding_box = [(0, 0), (0, 0)]        
        self.rect = None
        self.image_path = None
        self.background_image_path = None
        self.replacement_color = None
        self.processed_image = None
        self.original_image = None
        self.background_image_loaded = False
        self.preview_image = None
        self.preview_image_max_width = 600
        self.preview_image_max_height = 400
        self.aspect_ratio = 1.0
        self.is_drawing_box = False

        self.start()

    def start(self):
        """Initialize the UI components."""
        self.root.geometry("800x600")
        self.root.configure(bg="white")

        # Load image button and Process button on same row
        self.top_button_frame = tk.Frame(self.root, bg="white")
        self.top_button_frame.pack(pady=10)
        self.load_button = tk.Button(self.top_button_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=10)
        self.process_button = tk.Button(self.top_button_frame, text="Process Image", command=self.process_image)
        self.process_button.pack(side=tk.LEFT, padx=10)
        self.process_button.config(state=tk.DISABLED)
        self.reset_button = tk.Button(self.top_button_frame, text="Reset", command=self.reset)
        self.reset_button.pack(side=tk.LEFT, padx=10)

        #selection from radio buttons
        self.selection_frame = tk.LabelFrame(self.root, bg="white", text='Controls', border=1)
        self.selection_frame.pack(pady=10)
        
        self.selection_var = tk.StringVar(value="transparent")
        self.selection_var.trace_add("write", self.radiobutton_selected)
        self.transparent_radio = tk.Radiobutton(self.selection_frame, text="Transparent Background", variable=self.selection_var, value="transparent", bg="white")
        self.transparent_radio.pack(side=tk.LEFT, padx=10)
        self.color_radio = tk.Radiobutton(self.selection_frame, text="Color Background", variable=self.selection_var, value="color", bg="white")
        self.color_radio.pack(side=tk.LEFT, padx=10)
        self.image_radio = tk.Radiobutton(self.selection_frame, text="Image Background", variable=self.selection_var, value="image", bg="white")
        self.image_radio.pack(side=tk.LEFT, padx=10)

        ## Operation configuration frame
        self.operation_frame = tk.LabelFrame(self.root, bg="white", text='Operation Options', border=1)
        self.operation_frame.pack(pady=10)
        self.color_button = tk.Button(self.operation_frame, text="Pick Color", command=self.pick_replacement_color, state=tk.DISABLED)
        self.color_button.pack(side=tk.LEFT, padx=10)
        self.image_button = tk.Button(self.operation_frame, text="Load Background Image", command=self.load_background_image, state=tk.DISABLED)
        self.image_button.pack(side=tk.LEFT, padx=10)
        self.reset_bounding_box_button = tk.Button(self.operation_frame, text="Reset Bounding Box", command=self.reset_bounding_box, state=tk.DISABLED)
        self.reset_bounding_box_button.pack(side=tk.LEFT, padx=10)
        
        # Canvas for image display
        self.canvas = tk.Canvas(self.root, bg="white", width=600, height=400)
        self.canvas.pack(pady=10)
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.canvas.bind("<Motion>", self.on_mouse_move)

    def reset_bounding_box(self):
        """Reset the bounding box."""
        if self.rect:
            self.canvas.delete(self.rect)
            self.rect = None
        self.bounding_box = [(0, 0), (0, 0)]
        self.set_button_states()

    def reset(self):
        """Reset the application state."""
        self.canvas.delete("all")
        self.bounding_box = [(0, 0), (0, 0)]
        self.rect = None
        self.image_path = None
        self.background_image_path = None
        self.replacement_color = None
        self.processed_image = None
        self.original_image = None
        self.background_image_loaded = False
        self.preview_image = None
        self.aspect_ratio = 1.0
        self.is_drawing_box = False

        # Reset button states
        self.set_button_states()

    def radiobutton_selected(self, *args):
        """Handle radio button selection changes."""
        selected_value = self.selection_var.get()
        self.process_button.config(state=tk.NORMAL)
        if selected_value == "color":
            self.color_button.config(state=tk.NORMAL)
            self.image_button.config(state=tk.DISABLED)
            if self.replacement_color is None or self.bounding_box[1] == (0, 0):
                self.process_button.config(state=tk.DISABLED)
        elif selected_value == "image":
            self.color_button.config(state=tk.DISABLED)
            self.image_button.config(state=tk.NORMAL)
            if self.background_image_path is None or self.bounding_box[1] == (0, 0):
                self.process_button.config(state=tk.DISABLED)
        else:
            self.color_button.config(state=tk.DISABLED)
            self.image_button.config(state=tk.DISABLED)
            if self.bounding_box[1] == (0, 0):
                self.process_button.config(state=tk.DISABLED)
            

    def resize_preview_image(self, image: Image.Image) -> Image.Image:
        """
        Resize the image to fit within the maximum dimensions while maintaining aspect ratio.
        """
        width, height = image.size
        if width > self.preview_image_max_width or height > self.preview_image_max_height:
            self.aspect_ratio = width / height
            if self.aspect_ratio > 1:
                new_width = self.preview_image_max_width
                new_height = int(self.preview_image_max_width / self.aspect_ratio)
            else:
                new_height = self.preview_image_max_height
                new_width = int(self.preview_image_max_height * self.aspect_ratio)
            return image.resize((new_width, new_height), resample=Image.LANCZOS)
        return image
    
    def adjust_bounding_box_original_image(self):
        """
        Adjust the bounding box coordinates to expand to the aspect ratio of the original image.
        """
        if self.original_image is not None:
            # translate bounding box coordinates to coordinates on the original image
            x1, y1 = self.bounding_box[0]
            x2, y2 = self.bounding_box[1]
            original_width, original_height = self.original_image.shape[1], self.original_image.shape[0]
            
            x1 = int(x1 * original_width / self.preview_image.width())
            y1 = int(y1 * original_height / self.preview_image.height())
            x2 = int(x2 * original_width / self.preview_image.width())
            y2 = int(y2 * original_height / self.preview_image.height())

            self.bounding_box = [(x1, y1), (x2, y2)]


    def load_image(self):
        """
        Load an image from file. When the Load Image button is clicked, open a file dialog and save results to self.image_path.
        Reset the bounding box and processed image.
        Set the state of the buttons based on the selected radio button.
        """
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(self.image_path)
            self.canvas.delete("all")
            temp_img = Image.open(self.image_path)
            temp_img = ImageOps.exif_transpose(temp_img)  # Correct orientation based on EXIF data
            temp_img = self.resize_preview_image(temp_img)
            self.preview_image = ImageTk.PhotoImage(temp_img)            
            self.canvas.config(width=self.preview_image.width(), height=self.preview_image.height())
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.preview_image)


            self.set_button_states()            

            self.process_button.config(state=tk.NORMAL)

    def is_bounding_box_valid(self):
        """Check if the bounding box is valid."""
        return self.bounding_box[0] != (0, 0) and self.bounding_box[1] != (0, 0)
            
    def set_button_states(self):
        """Set the state of buttons based on the selected radio button."""
        self.reset_bounding_box_button.config(state=tk.NORMAL)
        if self.image_path is None:
            self.color_button.config(state=tk.DISABLED)
            self.image_button.config(state=tk.DISABLED)
            self.process_button.config(state=tk.DISABLED)
            self.reset_bounding_box_button.config(state=tk.DISABLED)
            return
        elif self.selection_var.get() == "color":
            self.color_button.config(state=tk.NORMAL)
            self.image_button.config(state=tk.DISABLED)
            if self.replacement_color is None or not self.is_bounding_box_valid():
                self.process_button.config(state=tk.DISABLED)
            else:
                self.process_button.config(state=tk.NORMAL)
        elif self.selection_var.get() == "image":
            self.color_button.config(state=tk.DISABLED)
            self.image_button.config(state=tk.NORMAL)
            if self.background_image_path is None or not self.is_bounding_box_valid():
                self.process_button.config(state=tk.DISABLED)
            else:
                self.process_button.config(state=tk.NORMAL)
        else:            
            self.color_button.config(state=tk.DISABLED)
            self.image_button.config(state=tk.DISABLED)
            if not self.is_bounding_box_valid():
                self.process_button.config(state=tk.DISABLED)
            else:
                self.process_button.config(state=tk.NORMAL)
        


    def load_background_image(self):
        """Load a background image from file."""
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.background_image_path = file_path
            self.background_image_loaded = True
            self.set_button_states()
            print(f"Loaded background image: {self.background_image_path}")

    def pick_replacement_color(self):
        """Open a color picker dialog and get replacement color."""
        color = colorchooser.askcolor(title="Choose Replacement Color")
        if color[0] is not None:
            self.replacement_color = tuple(int(c) for c in color[0])
            self.set_button_states()
            print(f"Selected color: {self.replacement_color}")

    def on_button_press(self, event):
        """Handle mouse button press events to select bounding box."""
        if self.rect:
            self.canvas.delete(self.rect)
            self.rect = None

        self.bounding_box = [self.get_bounding_box_coords(event.x, event.y), self.get_bounding_box_coords(event.x, event.y)]    
        self.create_rectangle()
        self.process_button.config(state=tk.DISABLED)
        self.set_button_states()
        self.is_drawing_box = True

    def get_bounding_box_coords(self, x, y):
        """Get the coordinates of the bounding box."""
        if x < 0:
            x = 0
        elif x > self.preview_image.width():
            x = self.preview_image.width()
        if y < 0:
            y = 0
        elif y > self.preview_image.height():
            y = self.preview_image.height()
        return (x, y)

    def create_rectangle(self):
        """Create a rectangle on the canvas."""
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.bounding_box[0][0], self.bounding_box[0][1], self.bounding_box[1][0], self.bounding_box[1][1], width=4, outline="red", tags="box")
            
        

    def on_button_release(self, event):
        """Handle mouse button release events to finalize bounding box."""
        if self.rect:
            self.canvas.delete(self.rect)
            self.bounding_box[1] = self.get_bounding_box_coords(event.x, event.y)
            self.create_rectangle()
            self.process_button.config(state=tk.NORMAL)
            self.set_button_states()
        self.is_drawing_box = False
        print(self.bounding_box)

    def on_mouse_move(self, event):
        """Handle mouse movement events to update bounding box and render on the UI."""
        if self.is_drawing_box:
            if self.rect:
                self.canvas.delete(self.rect)
                self.bounding_box[1] = self.get_bounding_box_coords(event.x, event.y)
            self.create_rectangle()
            

    def process_image(self):
        """Process the image to remove or replace background based on user selections."""
        if self.image_path is None or not self.is_bounding_box_valid():
            messagebox.showerror("Error", "Please load an image and select a bounding box.")
            return
        self.adjust_bounding_box_original_image()
        if self.selection_var.get() == "transparent":
            self.processed_image = self.transparent_processor.process_image(self.image_path, self.bounding_box)
        elif self.selection_var.get() == "color":
            if self.replacement_color is None:
                messagebox.showerror("Error", "Please select a replacement color.")
                return
            self.processed_image = self.colorbg_processor.process_image(self.image_path, self.bounding_box, self.replacement_color)
        elif self.selection_var.get() == "image":
            if not self.background_image_loaded:
                messagebox.showerror("Error", "Please load a background image.")
                return
            self.processed_image = self.img_processor.process_image(self.image_path, self.bounding_box, self.background_image_path)

        if self.processed_image is not None:
            cv2.imshow("Processed Image", self.processed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # Save the processed image to a file
            output_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if output_path:
                cv2.imwrite(output_path, self.processed_image)
                messagebox.showinfo("Success", f"Processed image saved to {output_path}.")
        else:
            messagebox.showerror("Error", "Failed to process the image.")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Background Replacement Tool")
    app_ui = AppUI(root)
    root.mainloop()
