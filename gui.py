import tkinter as tk
from tkinter import messagebox, ttk
import keyboard
import os
import sys

def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return relative_path

class SoundDetectorGUI:
    def __init__(self, state, controller):
        self.state = state
        self.controller = controller

        self.window = tk.Tk()
        self.pixel = tk.PhotoImage(width=1, height=1)

        self._build()
        self._hotkeyThread()

        self.window.mainloop()

    def _hotkeyThread(self):
        def on_key(event):
            if event.name == self.state.hotkeys["Start"]:
                self.controller.start()
            elif event.name == self.state.hotkeys["Stop"]:
                self.controller.stop()

        keyboard.on_press(on_key)

    def waitForSingleKey(callback):
        handle = None

        def on_key(event):
            nonlocal handle
            keyboard.unhook(handle)
            callback(event.name)
    
        handle = keyboard.on_press(on_key)

    def changeHotkey(self, key_name, tk_button, tk_label):
        tk_label.config(text=key_name + " (Press a key...)")

        def assignHotkey(key_pressed):
            for key in self.state.hotkeys:
                if key != key_name and self.state.hotkeys[key] == key_pressed:
                    messagebox.showerror(
                        title="Change failed!", 
                        message=key_pressed.upper() + " is already being used."
                    )
                    tk_label.config(text=key_name)
                    return

            self.state.hotkeys[key_name] = key_pressed
            tk_button.config(text = key_pressed.upper())
            tk_label.config(text = key_name)
            
        SoundDetectorGUI.waitForSingleKey(assignHotkey)
    
    def buildHotkeyWindow(self):
        newWindow = tk.Toplevel(self.window)
        newWindow.geometry("300x250")
        newWindow.title("Hotkeys")
        newWindow.resizable(False, False)

        # Subtitle
        tk.Label(
            newWindow, 
            text="Change hotkeys", 
            font=("Lexend", 12)
        ).grid(row=0, column=0, padx=5, sticky="W", pady=5)

        startLabel = tk.Label(
            newWindow, 
            text="Start", 
            font=("Lexend", 10), 
            image=self.pixel, 
            width=300 // 2 - 15, 
            height=30, 
            compound="center", 
            relief="groove", 
            borderwidth=1
        )
        startLabel.grid(row=1, column=0, padx=5, pady=5)
        
        startKey = tk.Button(
            newWindow, 
            text=self.state.hotkeys["Start"].upper(), 
            font=("Lexend", 10), 
            image=self.pixel, 
            width=300 // 2 - 15, 
            height=30, 
            compound="center", 
            relief="groove", 
            borderwidth=3, 
            command=lambda: self.changeHotkey("Start", startKey, startLabel)
        )
        startKey.grid(row=1, column=1, padx=5, pady=5)

        stopLabel = tk.Label(
            newWindow, 
            text="Stop", 
            font=("Lexend", 10), 
            image=self.pixel, 
            width=300 // 2 - 15, 
            height=30, 
            compound="center", 
            relief="groove", 
            borderwidth=1
        )
        stopLabel.grid(row=2, column=0, padx=5, pady=5)

        stopKey = tk.Button(
            newWindow, 
            text=self.state.hotkeys["Stop"].upper(), 
            font=("Lexend", 10), 
            image=self.pixel, 
            width=300 // 2 - 15, 
            height=30, 
            compound="center", 
            relief="groove", 
            borderwidth=3, 
            command=lambda: self.changeHotkey("Stop", stopKey, stopLabel)
        )
        stopKey.grid(row=2, column=1, padx=5, pady=5)
    
    def _build(self):
        self.window.geometry("400x350")
        self.window.title("Shiny Detector")
        self.window.resizable(False, False)

        icon = tk.PhotoImage(
            file=resource_path("assets/volume-icon.png")
        )
        self.window.iconphoto(True, icon)

        # Creates top frame
        frame1 = tk.Frame(self.window, highlightbackground="grey", highlightthickness=1)
        frame1.pack(side="top", padx=5, pady=5)

        # Subtitle
        tk.Label(
            frame1,
            text="Adjust settings",
            font=("Lexend", 8, "bold")
        ).grid(row=0, column=0, padx=5, sticky="w")

        # Chroma Slider
        tk.Label(
            frame1, 
            text="Chroma", 
            font=("Lexend", 8)
        ).grid(row=1, column=0, padx=5, sticky="w")

        self.state.chromaScale = tk.Scale(
            frame1,
            from_=0.5, 
            to=1, 
            length=400 - 10 - 30, 
            orient=tk.HORIZONTAL, 
            tickinterval=0.1, 
            resolution=0.02, 
            width=8, 
            activebackground="light blue", 
            sliderrelief="ridge", 
            borderwidth=2
        )
        self.state.chromaScale.set(0.9)
        self.state.chromaScale.grid(row=2, column=0, padx=10)

        # Combined Slider
        tk.Label(
            frame1, 
            text="Threshold", 
            font=("Lexend", 8)
        ).grid(row=3, column=0, padx=5, sticky="w")

        self.state.combScale = tk.Scale(
            frame1,
            from_=0.3, 
            to=1, 
            length=400 - 10 - 30, 
            orient=tk.HORIZONTAL, 
            tickinterval=0.1, 
            resolution=0.02, 
            width=8, 
            activebackground="light blue", 
            sliderrelief="ridge", 
            borderwidth=2
        )
        self.state.combScale.set(0.38)
        self.state.combScale.grid(row=4, column=0, padx=10)

        ttk.Separator(self.window, orient="horizontal").pack(fill="x", padx=5, pady=5)

        # Creates bottom-left frame
        frame2 = tk.Frame(self.window)
        frame2.pack(side="left", padx=5)

        tk.Button(
            frame2, 
            text="Hotkeys", 
            font=("Lexend", 10), 
            image=self.pixel, 
            width=400 // 2 - 5 - 5 - 10 - 10 - 10, 
            height=40, 
            compound="center", 
            relief="groove", 
            borderwidth=3,
            command=self.buildHotkeyWindow
        ).pack(side="top", padx=10, pady=5)

        tk.Button(
            frame2, 
            text="Help", 
            font=("Lexend", 10), 
            image=self.pixel, 
            width=400 // 2 - 5 - 5 - 10 - 10 - 10, 
            height=40, 
            compound="center", 
            relief="groove", 
            borderwidth=3
        ).pack(side="top", padx=10, pady=5)

        # Creates bottom-right frame
        frame3 = tk.Frame(self.window)
        frame3.pack(side="right", padx=5)

        tk.Button(
            frame3, 
            text="Start", 
            font=("Lexend", 10), 
            image=self.pixel, 
            width=400 // 2 - 5 - 5 - 10 - 10 - 10, 
            height=40, 
            compound="center", 
            relief="groove", 
            borderwidth=3,
            command=self.controller.start
        ).pack(side="top", padx=10, pady=5)

        tk.Button(
            frame3, 
            text="Stop", 
            font=("Lexend", 10), 
            image=self.pixel, 
            width=400 // 2 - 5 - 5 - 10 - 10 - 10, 
            height=40, 
            compound="center", 
            relief="groove", 
            borderwidth=3,
            command=self.controller.stop
        ).pack(side="top", padx=10, pady=5)