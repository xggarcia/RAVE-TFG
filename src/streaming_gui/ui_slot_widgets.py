import tkinter as tk
from tkinter import ttk


class StreamGUISlotWidgetsMixin:
    def create_model_slot_ui(self, slot):
        """Create a complete model slot with all controls."""
        slot_id = slot.slot_id
        colors = ["#3498db", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#8e44ad", "#34495e"]
        color_index = slot_id % len(colors)

        slot_frame = tk.LabelFrame(
            self.slots_container,
            text=f"Model Slot {slot_id + 1}",
            font=("Arial", 10, "bold"),
            bg="white",
            relief=tk.RIDGE,
            borderwidth=3,
            fg=colors[color_index],
        )
        slot_frame.pack(fill=tk.X, pady=5, padx=5)
        slot.frame = slot_frame

        top_row = tk.Frame(slot_frame, bg="white")
        top_row.pack(fill=tk.X, padx=10, pady=5)

        active_check = tk.Checkbutton(
            top_row,
            text="ACTIVE",
            variable=slot.is_active,
            font=("Arial", 9, "bold"),
            bg="white",
            fg=colors[color_index],
            command=lambda s=slot: self.toggle_slot_obj(s),
        )
        active_check.pack(side=tk.LEFT, padx=(0, 15))

        tk.Label(top_row, text="Model:", bg="white", width=6, anchor="w").pack(side=tk.LEFT)
        model_combo = ttk.Combobox(top_row, textvariable=slot.model_var, state="readonly", width=30)
        model_combo.pack(side=tk.LEFT, padx=5)
        slot.combo = model_combo

        load_btn = tk.Button(
            top_row,
            text="Load",
            command=lambda s=slot: self.load_model_to_slot_obj(s),
            bg=colors[color_index],
            fg="white",
            font=("Arial", 9, "bold"),
            width=8,
        )
        load_btn.pack(side=tk.LEFT, padx=5)

        browse_btn = tk.Button(
            top_row,
            text="Browse...",
            command=lambda s=slot: self.browse_model_for_slot_obj(s),
            width=10,
        )
        browse_btn.pack(side=tk.LEFT, padx=5)

        delete_btn = tk.Button(
            top_row,
            text="🗑",
            command=lambda s=slot: self.delete_model_slot(s),
            bg="#e74c3c",
            fg="white",
            font=("Arial", 10, "bold"),
            width=3,
            cursor="hand2",
        )
        delete_btn.pack(side=tk.RIGHT, padx=5)

        status_label = tk.Label(
            top_row,
            textvariable=slot.status_var,
            font=("Arial", 8, "italic"),
            bg="white",
            fg="gray",
        )
        status_label.pack(side=tk.LEFT, padx=10)

        input_row = tk.Frame(slot_frame, bg="white")
        input_row.pack(fill=tk.X, padx=10, pady=(5, 0))

        tk.Label(input_row, text="Input:", bg="white", font=("Arial", 8, "bold")).pack(side=tk.LEFT, padx=(0, 5))

        tk.Radiobutton(
            input_row,
            text="Random",
            variable=slot.input_mode,
            value="random",
            bg="white",
            command=lambda s=slot: self.update_input_controls_obj(s),
        ).pack(side=tk.LEFT, padx=5)

        tk.Radiobutton(
            input_row,
            text="Audio File",
            variable=slot.input_mode,
            value="audio",
            bg="white",
            command=lambda s=slot: self.update_input_controls_obj(s),
        ).pack(side=tk.LEFT, padx=5)

        slot.intensity_frame = tk.Frame(input_row, bg="white")
        tk.Label(slot.intensity_frame, text="Intensity:", bg="white", width=8).pack(side=tk.LEFT)
        tk.Scale(
            slot.intensity_frame,
            from_=0.1,
            to=3.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=slot.random_intensity,
            bg="white",
            length=100,
            showvalue=True,
        ).pack(side=tk.LEFT)
        slot.intensity_frame.pack(side=tk.LEFT, padx=10)

        slot.audio_frame = tk.Frame(input_row, bg="white")
        tk.Button(slot.audio_frame, text="Select Audio", command=lambda s=slot: self.select_audio_for_slot_obj(s), width=12).pack(
            side=tk.LEFT, padx=5
        )
        tk.Checkbutton(slot.audio_frame, text="Loop", variable=slot.loop_audio, bg="white").pack(side=tk.LEFT)

        self.update_input_controls_obj(slot)

        params_row = tk.Frame(slot_frame, bg="white")
        params_row.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.create_slot_slider(params_row, "Gain:", slot.gain, 0.0, 1.0, 0)
        self.create_slot_slider(params_row, "Temp:", slot.temperature, 0.1, 3.0, 1)
        self.create_slot_slider(params_row, "Smooth:", slot.smoothing, 0.0, 0.95, 2)

    @staticmethod
    def create_slot_slider(parent, label, variable, min_val, max_val, column):
        frame = tk.Frame(parent, bg="white")
        frame.grid(row=0, column=column, padx=10, pady=5, sticky="ew")

        tk.Label(frame, text=label, bg="white", width=6, anchor="w").pack(side=tk.LEFT)

        slider = tk.Scale(
            frame,
            from_=min_val,
            to=max_val,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=variable,
            bg="white",
            length=120,
            showvalue=True,
        )
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        parent.grid_columnconfigure(column, weight=1)
