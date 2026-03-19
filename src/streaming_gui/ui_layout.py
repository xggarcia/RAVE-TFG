import tkinter as tk
from tkinter import ttk


class StreamGUILayoutMixin:
    def create_widgets(self):
        """Create all GUI widgets."""
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)

        title_label = tk.Label(
            title_frame,
            text="RAVE Multi-Model Streaming",
            font=("Arial", 18, "bold"),
            bg="#2c3e50",
            fg="white",
        )
        title_label.pack(pady=15)

        main_container = tk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        canvas = tk.Canvas(main_container)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        content = scrollable_frame

        global_frame = ttk.LabelFrame(content, text="Global Audio Settings", padding=10)
        global_frame.pack(fill=tk.X, pady=(0, 15))

        sr_frame = tk.Frame(global_frame)
        sr_frame.pack(fill=tk.X, pady=5)
        tk.Label(sr_frame, text="Sample Rate:", width=15, anchor="w").pack(side=tk.LEFT)
        sr_combo = ttk.Combobox(
            sr_frame,
            textvariable=self.sr,
            values=[22050, 44100, 48000],
            state="readonly",
            width=15,
        )
        sr_combo.pack(side=tk.LEFT, padx=5)

        chunk_frame = tk.Frame(global_frame)
        chunk_frame.pack(fill=tk.X, pady=5)
        tk.Label(chunk_frame, text="Chunk Duration (s):", width=15, anchor="w").pack(side=tk.LEFT)
        chunk_spin = tk.Spinbox(
            chunk_frame,
            from_=0.5,
            to=3.0,
            increment=0.5,
            textvariable=self.chunk_duration,
            width=15,
        )
        chunk_spin.pack(side=tk.LEFT, padx=5)

        perf_frame = tk.Frame(global_frame)
        perf_frame.pack(fill=tk.X, pady=5)
        tk.Label(perf_frame, text="Performance Mode:", width=15, anchor="w").pack(side=tk.LEFT)
        perf_combo = ttk.Combobox(
            perf_frame,
            textvariable=self.performance_mode,
            values=["Quality", "Balanced", "Max Stability"],
            state="readonly",
            width=15,
        )
        perf_combo.pack(side=tk.LEFT, padx=5)
        tk.Label(
            perf_frame,
            text="Higher stability keeps audio running by reducing update rate under load",
            fg="gray",
        ).pack(side=tk.LEFT, padx=10)

        calibration_frame = tk.Frame(global_frame)
        calibration_frame.pack(fill=tk.X, pady=5)
        tk.Label(calibration_frame, text="Calibration:", width=15, anchor="w").pack(side=tk.LEFT)
        self.calibrate_button = tk.Button(
            calibration_frame,
            text="Quick Calibrate",
            command=self.run_quick_calibration,
            width=15,
        )
        self.calibrate_button.pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(
            calibration_frame,
            text="Auto-calibrate before streaming",
            variable=self.auto_calibrate_on_start,
        ).pack(side=tk.LEFT, padx=10)

        calibration_status_frame = tk.Frame(global_frame)
        calibration_status_frame.pack(fill=tk.X, pady=2)
        tk.Label(
            calibration_status_frame,
            textvariable=self.calibration_summary,
            fg="gray",
            anchor="w",
            justify=tk.LEFT,
        ).pack(side=tk.LEFT, padx=(15, 0))

        slots_header = ttk.LabelFrame(content, text="Model Slots", padding=10)
        slots_header.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        add_button_frame = tk.Frame(slots_header)
        add_button_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Button(
            add_button_frame,
            text="+ Add Model Slot",
            command=self.add_model_slot,
            bg="#27ae60",
            fg="white",
            font=("Arial", 10, "bold"),
            cursor="hand2",
        ).pack(side=tk.LEFT, padx=5)

        tk.Label(
            add_button_frame,
            text="(Activate multiple to mix audio)",
            font=("Arial", 9, "italic"),
            fg="gray",
        ).pack(side=tk.LEFT, padx=10)

        self.slots_container = tk.Frame(slots_header)
        self.slots_container.pack(fill=tk.BOTH, expand=True)

        status_frame = ttk.LabelFrame(content, text="Status Log", padding=10)
        status_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        self.status_text = tk.Text(
            status_frame,
            height=8,
            width=70,
            state="disabled",
            bg="#f0f0f0",
            font=("Consolas", 9),
        )
        self.status_text.pack(fill=tk.BOTH, expand=True)

        status_scroll = ttk.Scrollbar(status_frame, command=self.status_text.yview)
        status_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.config(yscrollcommand=status_scroll.set)

        self.add_model_slot()

        button_frame = tk.Frame(content)
        button_frame.pack(fill=tk.X, pady=10)

        self.start_button = tk.Button(
            button_frame,
            text="START STREAMING",
            command=self.start_streaming,
            bg="#27ae60",
            fg="white",
            font=("Arial", 12, "bold"),
            height=2,
            cursor="hand2",
        )
        self.start_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.stop_button = tk.Button(
            button_frame,
            text="STOP STREAMING",
            command=self.stop_streaming,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 12, "bold"),
            height=2,
            state="disabled",
            cursor="hand2",
        )
        self.stop_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))

        self.log_status("GUI initialized. Add model slots to begin.")
