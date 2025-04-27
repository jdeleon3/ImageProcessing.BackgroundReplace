import tkinter as tk
from appui import AppUI

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Background Replacement Tool")
    

    app_ui = AppUI(root)
    root.mainloop()