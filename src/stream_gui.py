# -*- coding: utf-8 -*-
"""
GUI for Real-time RAVE Audio Streaming - Multi-Model Support
Interactive visual controls for multiple models with independent parameters
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import torch
import numpy as np
import sounddevice as sd
import librosa

torch.set_grad_enabled(False)


class ModelSlot:
    """Represents a single model slot with its own parameters and state"""
    def __init__(self, slot_id):
        self.slot_id = slot_id
        self.model = None
        self.model_path = None
        self.is_active = tk.BooleanVar(value=False)
        self.is_loaded = False
        
        # Model-specific dimensions
        self.latent_size = None
        self.latent_length = None
        self.output_length = None  # Length of decoded audio output
        self.prev_z = None
        
        # Individual parameters for this model
        self.gain = tk.DoubleVar(value=0.5)  # Lower default for mixing
        self.temperature = tk.DoubleVar(value=1.0)
        self.smoothing = tk.DoubleVar(value=0.0)
        
        # Input source parameters
        self.input_mode = tk.StringVar(value="random")  # "random" or "audio"
        self.audio_file_path = None
        self.encoded_latents = None  # Encoded audio latents
        self.latent_position = 0  # Current position in encoded latents
        self.loop_audio = tk.BooleanVar(value=True)
        self.random_intensity = tk.DoubleVar(value=1.0)  # Scale for random noise
        
        # UI references
        self.model_var = tk.StringVar(value="[No Model Loaded]")
        self.status_var = tk.StringVar(value="Inactive")


class RAVEStreamGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RAVE Multi-Model Streaming")
        self.root.geometry("1000x850")
        self.root.resizable(True, True)
        
        # Streaming state
        self.is_streaming = False
        self.stream = None
        self.stream_thread = None
        
        # Global audio settings
        self.sr = tk.IntVar(value=44100)
        self.chunk_duration = tk.DoubleVar(value=1.0)
        self.chunk_samples = None
        
        # Dynamic model slots (start with 1)
        self.model_slots = []
        self.next_slot_id = 0
        self.slots_container = None  # Will hold the slots frame
        
        # Available models list
        self.available_models = []
        
        # Create GUI
        self.create_widgets()
        
        # Find available models
        self.discover_models()
    
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="RAVE Multi-Model Streaming", 
            font=('Arial', 18, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=15)
        
        # Main container with scrollbar
        main_container = tk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas and scrollbar for scrollable content
        canvas = tk.Canvas(main_container)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        content = scrollable_frame
        
        # Global Audio Settings
        global_frame = ttk.LabelFrame(content, text="Global Audio Settings", padding=10)
        global_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Sample rate
        sr_frame = tk.Frame(global_frame)
        sr_frame.pack(fill=tk.X, pady=5)
        tk.Label(sr_frame, text="Sample Rate:", width=15, anchor='w').pack(side=tk.LEFT)
        sr_combo = ttk.Combobox(sr_frame, textvariable=self.sr, 
                                values=[22050, 44100, 48000], 
                                state="readonly", width=15)
        sr_combo.pack(side=tk.LEFT, padx=5)
        
        # Chunk duration
        chunk_frame = tk.Frame(global_frame)
        chunk_frame.pack(fill=tk.X, pady=5)
        tk.Label(chunk_frame, text="Chunk Duration (s):", width=15, anchor='w').pack(side=tk.LEFT)
        chunk_spin = tk.Spinbox(chunk_frame, from_=0.5, to=3.0, increment=0.5,
                                textvariable=self.chunk_duration, width=15)
        chunk_spin.pack(side=tk.LEFT, padx=5)
        
        # Model Slots Section
        slots_header = ttk.LabelFrame(content, text="Model Slots", padding=10)
        slots_header.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Add Model button
        add_button_frame = tk.Frame(slots_header)
        add_button_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Button(
            add_button_frame,
            text="+ Add Model Slot",
            command=self.add_model_slot,
            bg='#27ae60',
            fg='white',
            font=('Arial', 10, 'bold'),
            cursor='hand2'
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Label(
            add_button_frame,
            text="(Activate multiple to mix audio)",
            font=('Arial', 9, 'italic'),
            fg='gray'
        ).pack(side=tk.LEFT, padx=10)
        
        # Slots container (scrollable)
        self.slots_container = tk.Frame(slots_header)
        self.slots_container.pack(fill=tk.BOTH, expand=True)
        
        # Status Log (create BEFORE adding first slot so log_status works)
        status_frame = ttk.LabelFrame(content, text="Status Log", padding=10)
        status_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        self.status_text = tk.Text(status_frame, height=8, width=70, 
                                   state='disabled', bg='#f0f0f0',
                                   font=('Consolas', 9))
        self.status_text.pack(fill=tk.BOTH, expand=True)
        
        status_scroll = ttk.Scrollbar(status_frame, command=self.status_text.yview)
        status_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.config(yscrollcommand=status_scroll.set)
        
        # Create first slot by default (after status_text is initialized)
        self.add_model_slot()
        
        # Control Buttons
        button_frame = tk.Frame(content)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.start_button = tk.Button(
            button_frame,
            text="START STREAMING",
            command=self.start_streaming,
            bg='#27ae60',
            fg='white',
            font=('Arial', 12, 'bold'),
            height=2,
            cursor='hand2'
        )
        self.start_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.stop_button = tk.Button(
            button_frame,
            text="STOP STREAMING",
            command=self.stop_streaming,
            bg='#e74c3c',
            fg='white',
            font=('Arial', 12, 'bold'),
            height=2,
            state='disabled',
            cursor='hand2'
        )
        self.stop_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        
        self.log_status("GUI initialized. Add model slots to begin.")
    
    def add_model_slot(self):
        """Add a new model slot dynamically"""
        if self.is_streaming:
            messagebox.showwarning(
                "Cannot Add Slot",
                "Please stop streaming before adding new slots."
            )
            return
        
        # Create new slot with unique ID
        slot = ModelSlot(self.next_slot_id)
        self.model_slots.append(slot)
        
        # Create the slot UI
        self.create_model_slot_ui(slot)
        
        # Update dropdowns for all slots
        self.update_all_dropdowns()
        
        self.log_status(f"Added Slot {self.next_slot_id + 1}")
        self.next_slot_id += 1
    
    def delete_model_slot(self, slot):
        """Delete a model slot"""
        if self.is_streaming:
            messagebox.showwarning(
                "Cannot Delete Slot",
                "Please stop streaming before deleting slots."
            )
            return
        
        if len(self.model_slots) <= 1:
            messagebox.showwarning(
                "Cannot Delete",
                "You must have at least one slot."
            )
            return
        
        # Confirm deletion
        if messagebox.askyesno(
            "Delete Slot",
            f"Delete Slot {slot.slot_id + 1}?\n\nThis will remove the model and all its settings."
        ):
            # Remove from list
            self.model_slots.remove(slot)
            
            # Destroy the UI frame
            if hasattr(slot, 'frame'):
                slot.frame.destroy()
            
            self.log_status(f"Deleted Slot {slot.slot_id + 1}")
    
    def create_model_slot_ui(self, slot):
        """Create a complete model slot with all controls"""
        slot_id = slot.slot_id
        
        # Slot container with colored border
        colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#8e44ad', '#34495e']
        color_index = slot_id % len(colors)
        
        slot_frame = tk.LabelFrame(
            self.slots_container, 
            text=f"Model Slot {slot_id + 1}", 
            font=('Arial', 10, 'bold'),
            bg='white',
            relief=tk.RIDGE,
            borderwidth=3,
            fg=colors[color_index]
        )
        slot_frame.pack(fill=tk.X, pady=5, padx=5)
        slot.frame = slot_frame  # Save reference for deletion
        
        # Top row: Active checkbox + Model selection + Load button + Delete button
        top_row = tk.Frame(slot_frame, bg='white')
        top_row.pack(fill=tk.X, padx=10, pady=5)
        
        # Active checkbox
        active_check = tk.Checkbutton(
            top_row,
            text="ACTIVE",
            variable=slot.is_active,
            font=('Arial', 9, 'bold'),
            bg='white',
            fg=colors[color_index],
            command=lambda s=slot: self.toggle_slot_obj(s)
        )
        active_check.pack(side=tk.LEFT, padx=(0, 15))
        
        # Model selection
        tk.Label(top_row, text="Model:", bg='white', width=6, anchor='w').pack(side=tk.LEFT)
        model_combo = ttk.Combobox(
            top_row, 
            textvariable=slot.model_var,
            state="readonly", 
            width=30
        )
        model_combo.pack(side=tk.LEFT, padx=5)
        slot.combo = model_combo  # Save reference
        
        # Load button
        load_btn = tk.Button(
            top_row,
            text="Load",
            command=lambda s=slot: self.load_model_to_slot_obj(s),
            bg=colors[color_index],
            fg='white',
            font=('Arial', 9, 'bold'),
            width=8
        )
        load_btn.pack(side=tk.LEFT, padx=5)
        
        # Browse button
        browse_btn = tk.Button(
            top_row,
            text="Browse...",
            command=lambda s=slot: self.browse_model_for_slot_obj(s),
            width=10
        )
        browse_btn.pack(side=tk.LEFT, padx=5)
        
        # Delete button
        delete_btn = tk.Button(
            top_row,
            text="🗑",
            command=lambda s=slot: self.delete_model_slot(s),
            bg='#e74c3c',
            fg='white',
            font=('Arial', 10, 'bold'),
            width=3,
            cursor='hand2'
        )
        delete_btn.pack(side=tk.RIGHT, padx=5)
        
        # Status
        status_label = tk.Label(
            top_row,
            textvariable=slot.status_var,
            font=('Arial', 8, 'italic'),
            bg='white',
            fg='gray'
        )
        status_label.pack(side=tk.LEFT, padx=10)
        
        # Input Source Row
        input_row = tk.Frame(slot_frame, bg='white')
        input_row.pack(fill=tk.X, padx=10, pady=(5, 0))
        
        tk.Label(input_row, text="Input:", bg='white', font=('Arial', 8, 'bold')).pack(side=tk.LEFT, padx=(0, 5))
        
        # Radio buttons for input mode
        tk.Radiobutton(
            input_row,
            text="Random",
            variable=slot.input_mode,
            value="random",
            bg='white',
            command=lambda s=slot: self.update_input_controls_obj(s)
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Radiobutton(
            input_row,
            text="Audio File",
            variable=slot.input_mode,
            value="audio",
            bg='white',
            command=lambda s=slot: self.update_input_controls_obj(s)
        ).pack(side=tk.LEFT, padx=5)
        
        # Random intensity slider (shown when Random mode)
        slot.intensity_frame = tk.Frame(input_row, bg='white')
        tk.Label(slot.intensity_frame, text="Intensity:", bg='white', width=8).pack(side=tk.LEFT)
        tk.Scale(
            slot.intensity_frame,
            from_=0.1,
            to=3.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=slot.random_intensity,
            bg='white',
            length=100,
            showvalue=True
        ).pack(side=tk.LEFT)
        slot.intensity_frame.pack(side=tk.LEFT, padx=10)
        
        # Audio file controls (shown when Audio mode)
        slot.audio_frame = tk.Frame(input_row, bg='white')
        tk.Button(
            slot.audio_frame,
            text="Select Audio",
            command=lambda s=slot: self.select_audio_for_slot_obj(s),
            width=12
        ).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(
            slot.audio_frame,
            text="Loop",
            variable=slot.loop_audio,
            bg='white'
        ).pack(side=tk.LEFT)
        
        # Initially hide audio controls
        self.update_input_controls_obj(slot)
        
        # Parameters row
        params_row = tk.Frame(slot_frame, bg='white')
        params_row.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Gain slider
        self.create_slot_slider(params_row, "Gain:", slot.gain, 0.0, 1.0, 0)
        
        # Temperature slider
        self.create_slot_slider(params_row, "Temp:", slot.temperature, 0.1, 3.0, 1)
        
        # Smoothing slider
        self.create_slot_slider(params_row, "Smooth:", slot.smoothing, 0.0, 0.95, 2)
    
    def create_slot_slider(self, parent, label, variable, min_val, max_val, column):
        """Create a parameter slider for a model slot"""
        frame = tk.Frame(parent, bg='white')
        frame.grid(row=0, column=column, padx=10, pady=5, sticky='ew')
        
        tk.Label(frame, text=label, bg='white', width=6, anchor='w').pack(side=tk.LEFT)
        
        slider = tk.Scale(
            frame,
            from_=min_val,
            to=max_val,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=variable,
            bg='white',
            length=120,
            showvalue=True
        )
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Configure grid weights for even distribution
        parent.grid_columnconfigure(column, weight=1)
    
    def toggle_slot_obj(self, slot):
        """Handle slot activation/deactivation (object-based)"""
        if slot.is_active.get():
            if not slot.is_loaded:
                messagebox.showwarning(
                    "Model Not Loaded",
                    f"Please load a model in Slot {slot.slot_id + 1} before activating."
                )
                slot.is_active.set(False)
                return
            slot.status_var.set("Active")
            self.log_status(f"Slot {slot.slot_id + 1} activated: {os.path.basename(slot.model_path)}")
        else:
            slot.status_var.set("Loaded (Inactive)")
            self.log_status(f"Slot {slot.slot_id + 1} deactivated")
    
    def update_input_controls_obj(self, slot):
        """Show/hide input controls based on selected mode (object-based)"""
        if slot.input_mode.get() == "random":
            slot.intensity_frame.pack(side=tk.LEFT, padx=10)
            slot.audio_frame.pack_forget()
        else:  # audio mode
            slot.intensity_frame.pack_forget()
            slot.audio_frame.pack(side=tk.LEFT, padx=10)
    
    def update_input_controls(self, slot_id):
        """Show/hide input controls based on selected mode (deprecated, use object version)"""
        slot = self.model_slots[slot_id]
        self.update_input_controls_obj(slot)
    
    def select_audio_for_slot_obj(self, slot):
        """Select an audio file for a slot (object-based)"""
        filename = filedialog.askopenfilename(
            title=f"Select Audio File for Slot {slot.slot_id + 1}",
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.flac *.ogg"),
                ("WAV Files", "*.wav"),
                ("All Files", "*.*")
            ],
            initialdir=os.path.join(os.getcwd(), "input_data")
        )
        
        if not filename:
            return
        
        if not slot.is_loaded:
            messagebox.showwarning(
                "Model Not Loaded",
                "Please load a model first before selecting audio input."
            )
            return
        
        try:
            self.log_status(f"Slot {slot.slot_id + 1}: Loading audio file {os.path.basename(filename)}...")
            
            # Load audio file
            audio, sr_original = librosa.load(filename, sr=self.sr.get(), mono=True)
            
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
            
            # Encode audio to latent space
            with torch.no_grad():
                encoded = slot.model.encode(audio_tensor)
            
            # Store encoded latents
            slot.audio_file_path = filename
            slot.encoded_latents = encoded
            slot.latent_position = 0
            
            total_chunks = encoded.shape[-1] // slot.latent_length
            duration = len(audio) / self.sr.get()
            
            self.log_status(f"Slot {slot.slot_id + 1}: Audio encoded ({duration:.1f}s, {total_chunks} chunks)")
            
        except Exception as e:
            messagebox.showerror("Audio Load Error", f"Failed to load audio:\n{str(e)}")
            self.log_status(f"Slot {slot.slot_id + 1}: Audio load failed - {str(e)}")
    
    def select_audio_for_slot(self, slot_id):
        """Select an audio file for a slot (deprecated, use object version)"""
        slot = self.model_slots[slot_id]
        self.select_audio_for_slot_obj(slot)
    
    def create_slot_slider(self, parent, label, variable, min_val, max_val, column):
        """Create a parameter slider for a model slot"""
        frame = tk.Frame(parent, bg='white')
        frame.grid(row=0, column=column, padx=10, pady=5, sticky='ew')
        
        tk.Label(frame, text=label, bg='white', width=6, anchor='w').pack(side=tk.LEFT)
        
        slider = tk.Scale(
            frame,
            from_=min_val,
            to=max_val,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=variable,
            bg='white',
            length=120,
            showvalue=True
        )
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Configure grid weights for even distribution
        parent.grid_columnconfigure(column, weight=1)
    
    def toggle_slot(self, slot_id):
        """Handle slot activation/deactivation (deprecated, use object version)"""
        slot = self.model_slots[slot_id]
        self.toggle_slot_obj(slot)
    
    def load_model_to_slot_obj(self, slot):
        """Load selected model into a slot (object-based)"""
        model_path = slot.model_var.get()
        
        if model_path == "[No Model Loaded]" or not model_path:
            messagebox.showwarning("No Model", "Please select a model from the dropdown.")
            return
        
        # Find full path
        full_path = None
        for available_path in self.available_models:
            if os.path.basename(available_path) == model_path:
                full_path = available_path
                break
        
        if not full_path or not os.path.exists(full_path):
            messagebox.showerror("Error", f"Model file not found: {model_path}")
            return
        
        try:
            self.log_status(f"Loading {model_path} into Slot {slot.slot_id + 1}...")
            
            # Load model
            model = torch.jit.load(full_path).eval()
            
            # Detect latent dimensions by testing common configurations
            with torch.no_grad():
                # Test for latent size (channels) - try common values: 1, 2, 8, 16, 32, 64, 128
                latent_size = None
                for test_channels in [1, 2, 8, 16, 32, 64, 128]:
                    try:
                        test_z = torch.randn(1, test_channels, 128)  # Test with 128 length
                        _ = model.decode(test_z)
                        latent_size = test_channels
                        self.log_status(f"  Detected latent channels: {latent_size}")
                        break
                    except:
                        continue
                
                if latent_size is None:
                    raise Exception("Could not determine latent channel size")
                
                # Find minimum latent length and corresponding output size
                # Test powers of 2 from 1 to 16384
                latent_length = 1
                output_length = None
                max_latent = 16384
                
                while latent_length <= max_latent:
                    try:
                        test_z = torch.randn(1, latent_size, latent_length)
                        test_output = model.decode(test_z)
                        output_length = test_output.shape[-1]
                        self.log_status(f"  Detected: latent_size={latent_size}, latent_length={latent_length}, output={output_length}")
                        break
                    except Exception as test_error:
                        # Continue to next power of 2
                        latent_length *= 2
                
                if output_length is None:
                    raise Exception(f"Could not determine model dimensions (tested up to {max_latent})")
            
            # Store in slot
            slot.model = model
            slot.model_path = full_path
            slot.latent_size = latent_size
            slot.latent_length = latent_length
            slot.output_length = output_length  # Store the output size
            slot.prev_z = None
            slot.is_loaded = True
            slot.status_var.set("Loaded (Inactive)")
            
            self.log_status(f"Slot {slot.slot_id + 1}: Model loaded successfully")
            self.log_status(f"  Latent: {slot.latent_size}×{slot.latent_length}, Output: {slot.output_length} samples")
            
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load model:\n{str(e)}")
            self.log_status(f"Slot {slot.slot_id + 1}: Load failed - {str(e)}")
    
    def load_model_to_slot(self, slot_id):
        """Load selected model into a slot (deprecated, use object version)"""
        slot = self.model_slots[slot_id]
        self.load_model_to_slot_obj(slot)
    
    def browse_model_for_slot_obj(self, slot):
        """Browse for a model file (object-based)"""
        filename = filedialog.askopenfilename(
            title=f"Select Model for Slot {slot.slot_id + 1}",
            filetypes=[("TorchScript Models", "*.ts"), ("All Files", "*.*")],
            initialdir=os.path.join(os.getcwd(), "models")
        )
        
        if filename:
            slot.model_var.set(os.path.basename(filename))
            # Add to available models if not already there
            if filename not in self.available_models:
                self.available_models.append(filename)
                self.update_all_dropdowns()
    
    def browse_model_for_slot(self, slot_id):
        """Browse for a model file for specific slot (deprecated, use object version)"""
        slot = self.model_slots[slot_id]
        self.browse_model_for_slot_obj(slot)
    
    def update_all_dropdowns(self):
        """Update all slot dropdowns with available models"""
        model_names = ["[No Model Loaded]"] + [os.path.basename(m) for m in self.available_models]
        for slot in self.model_slots:
            if hasattr(slot, 'combo'):
                slot.combo['values'] = model_names
    
    def discover_models(self):
        """Find all .ts models in standard directories"""
        search_dirs = [
            os.path.join(os.getcwd(), "models", "demo_model"),
            os.path.join(os.getcwd(), "models", "user_model", "exported_model")
        ]
        
        self.available_models = []
        for directory in search_dirs:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    if file.endswith('.ts'):
                        full_path = os.path.join(directory, file)
                        self.available_models.append(full_path)
        
        self.update_all_dropdowns()
        self.log_status(f"Found {len(self.available_models)} available models")
    
    def log_status(self, message):
        """Add a message to the status log"""
        self.status_text.config(state='normal')
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.status_text.config(state='disabled')
    
    def start_streaming(self):
        """Start real-time audio streaming"""
        # Check if at least one model is active
        active_slots = [slot for slot in self.model_slots if slot.is_active.get() and slot.is_loaded]
        
        if not active_slots:
            messagebox.showwarning(
                "No Active Models",
                "Please load and activate at least one model before starting."
            )
            return
        
        # Use the output size from the first active model
        # (All models should produce compatible output sizes)
        self.chunk_samples = active_slots[0].output_length
        
        self.log_status("=" * 50)
        self.log_status(f"Starting streaming with {len(active_slots)} active model(s)")
        self.log_status(f"Sample Rate: {self.sr.get()} Hz")
        actual_duration = self.chunk_samples / self.sr.get()
        self.log_status(f"Chunk: {self.chunk_samples} samples ({actual_duration:.3f}s)")
        
        for slot in active_slots:
            input_info = "Random" if slot.input_mode.get() == "random" else f"Audio: {os.path.basename(slot.audio_file_path) if slot.audio_file_path else 'None'}"
            self.log_status(f"  Slot {slot.slot_id + 1}: {os.path.basename(slot.model_path)} [{input_info}]")
            self.log_status(f"    Gain={slot.gain.get():.2f}, Temp={slot.temperature.get():.2f}, Smooth={slot.smoothing.get():.2f}")
        
        # Initialize audio stream
        try:
            self.stream = sd.OutputStream(
                samplerate=self.sr.get(),
                channels=1,
                dtype='float32',
                blocksize=self.chunk_samples
            )
            self.stream.start()
            self.log_status("Audio stream initialized")
        except Exception as e:
            messagebox.showerror("Audio Error", f"Failed to initialize audio:\n{str(e)}")
            return
        
        # Start streaming thread
        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._streaming_thread, daemon=True)
        self.stream_thread.start()
        
        # Update UI
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        
        self.log_status("STREAMING STARTED - Mixing active models in real-time")
    
    def stop_streaming(self):
        """Stop the audio stream"""
        self.is_streaming = False
        
        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # Update UI
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
        self.log_status("STREAMING STOPPED")
    
    def _streaming_thread(self):
        """Background thread for audio generation with multi-model mixing"""
        try:
            while self.is_streaming:
                # Get currently active slots
                active_slots = [slot for slot in self.model_slots if slot.is_active.get() and slot.is_loaded]
                
                if not active_slots:
                    # No active models, output silence
                    silence = np.zeros(self.chunk_samples, dtype=np.float32)
                    self.stream.write(silence)
                    continue
                
                # Generate audio from each active model
                mixed_audio = np.zeros(self.chunk_samples, dtype=np.float32)
                
                for slot in active_slots:
                    with torch.no_grad():
                        # Get latent vector based on input mode
                        if slot.input_mode.get() == "audio" and slot.encoded_latents is not None:
                            # Audio file mode: use encoded latents
                            total_latent_frames = slot.encoded_latents.shape[-1]
                            
                            # Check if we have enough frames left
                            if slot.latent_position + slot.latent_length <= total_latent_frames:
                                # Extract the chunk
                                z = slot.encoded_latents[:, :, slot.latent_position:slot.latent_position + slot.latent_length]
                                slot.latent_position += slot.latent_length
                            else:
                                # End of audio
                                if slot.loop_audio.get():
                                    # Loop back to start
                                    slot.latent_position = 0
                                    z = slot.encoded_latents[:, :, slot.latent_position:slot.latent_position + slot.latent_length]
                                    slot.latent_position += slot.latent_length
                                else:
                                    # Audio finished, deactivate slot
                                    slot.is_active.set(False)
                                    slot.status_var.set("Loaded (Audio Finished)")
                                    self.log_status(f"Slot {slot.slot_id + 1}: Audio playback finished")
                                    continue
                            
                            # Apply temperature to encoded audio
                            z = z * slot.temperature.get()
                        else:
                            # Random mode: generate random latent vector
                            z = torch.randn(1, slot.latent_size, slot.latent_length)
                            # Apply intensity (random_intensity) and temperature
                            z = z * slot.random_intensity.get() * slot.temperature.get()
                        
                        # Apply smoothing
                        if slot.prev_z is not None:
                            smooth = slot.smoothing.get()
                            z = smooth * slot.prev_z + (1 - smooth) * z
                        slot.prev_z = z
                        
                        # Decode to get raw audio
                        audio = slot.model.decode(z).cpu().numpy().flatten()
                        
                        # Ensure audio matches expected output length
                        # (Should naturally match, but handle edge cases)
                        if len(audio) != self.chunk_samples:
                            if len(audio) > self.chunk_samples:
                                audio = audio[:self.chunk_samples]
                            else:
                                # If shorter, pad with zeros
                                audio = np.pad(audio, (0, self.chunk_samples - len(audio)))
                        
                        # Apply gain and add to mix
                        audio = audio * slot.gain.get()
                        mixed_audio += audio                        # Apply gain and add to mix
                        audio = audio * slot.gain.get()
                        mixed_audio += audio
                
                # Normalize mixed audio if multiple models
                if len(active_slots) > 1:
                    max_val = np.abs(mixed_audio).max()
                    if max_val > 1.0:
                        mixed_audio = mixed_audio / max_val
                
                # Ensure float32
                mixed_audio = mixed_audio.astype(np.float32)
                
                # Write to stream
                self.stream.write(mixed_audio)
                
        except Exception as e:
            self.log_status(f"ERROR in streaming: {str(e)}")
            self.is_streaming = False


def launch_gui():
    """Launch the GUI application"""
    # Ensure UTF-8 encoding for Windows
    if sys.platform == "win32":
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')
    
    root = tk.Tk()
    app = RAVEStreamGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
