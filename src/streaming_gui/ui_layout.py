import time
import tkinter as tk
import customtkinter as ctk


class StreamGUILayoutMixin:
    def create_widgets(self):
        """Create all GUI widgets using a mixing-desk inspired dark layout."""
        # Theme tokens inspired by tools/web.html reference.
        bg_root = "#161a20"
        desk_bg = "#323741"
        panel_bg = "#2b3038"
        panel_border = "#111111"
        text_main = "#d3dae3"
        text_muted = "#9da8b5"
        text_bright = "#d9dee5"
        accent_green = "#4caf50"
        accent_red = "#d32f2f"
        accent_orange = "#d97736"

        self.root.configure(fg_color=bg_root)

        desk = ctk.CTkFrame(
            self.root,
            fg_color=desk_bg,
            border_width=1,
            border_color=panel_border,
            corner_radius=16,
        )
        desk.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        header = ctk.CTkFrame(desk, fg_color="transparent")
        header.pack(fill=tk.X, padx=12, pady=(10, 8))
        ctk.CTkLabel(
            header,
            text="RAVE Multi-Model Streaming",
            font=("Segoe UI", 18, "bold"),
            text_color=text_bright,
        ).pack()

        body = ctk.CTkFrame(desk, fg_color="transparent")
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # Left panel: global settings.
        left_panel = ctk.CTkFrame(
            body,
            fg_color=panel_bg,
            corner_radius=14,
            border_width=1,
            border_color=panel_border,
        )
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        ctk.CTkLabel(
            left_panel,
            text="Global Settings",
            font=("Segoe UI", 13, "bold"),
            text_color=text_muted,
        ).pack(anchor="w", padx=12, pady=(10, 4))

        def row(parent, label):
            line = ctk.CTkFrame(parent, fg_color="transparent")
            line.pack(fill=tk.X, padx=10, pady=4)
            ctk.CTkLabel(
                line,
                text=label,
                width=120,
                anchor="w",
                text_color=text_main,
                font=("Segoe UI", 11),
            ).pack(side=tk.LEFT)
            return line

        sr_row = row(left_panel, "Sample Rate")
        sr_var = tk.StringVar(value=str(self.sr.get()))
        ctk.CTkOptionMenu(
            sr_row,
            values=["22050", "44100", "48000"],
            variable=sr_var,
            command=lambda value: self.sr.set(int(value)),
            width=120,
            corner_radius=8,
            fg_color="#3a424d",
            button_color="#2a3038",
            button_hover_color="#232830",
            dropdown_fg_color="#2a3038",
            text_color=text_main,
        ).pack(side=tk.RIGHT)

        chunk_row = row(left_panel, "Chunk Duration")
        chunk_var = tk.StringVar(value=str(self.chunk_duration.get()))
        ctk.CTkOptionMenu(
            chunk_row,
            values=["0.5", "1.0", "1.5", "2.0", "2.5", "3.0"],
            variable=chunk_var,
            command=lambda value: self.chunk_duration.set(float(value)),
            width=120,
            corner_radius=8,
            fg_color="#3a424d",
            button_color="#2a3038",
            button_hover_color="#232830",
            dropdown_fg_color="#2a3038",
            text_color=text_main,
        ).pack(side=tk.RIGHT)

        perf_row = row(left_panel, "Performance")
        ctk.CTkOptionMenu(
            perf_row,
            values=["Quality", "Balanced", "Max Stability"],
            variable=self.performance_mode,
            width=140,
            corner_radius=8,
            fg_color="#3a424d",
            button_color="#2a3038",
            button_hover_color="#232830",
            dropdown_fg_color="#2a3038",
            text_color=text_main,
        ).pack(side=tk.RIGHT)

        cal_row = row(left_panel, "Calibration")
        self.calibrate_button = ctk.CTkButton(
            cal_row,
            text="Quick Calibrate",
            command=self.run_quick_calibration,
            width=120,
            height=30,
            corner_radius=8,
            fg_color="#444444",
            hover_color="#555555",
            text_color=text_bright,
        )
        self.calibrate_button.pack(side=tk.RIGHT)

        auto_row = ctk.CTkFrame(left_panel, fg_color="transparent")
        auto_row.pack(fill=tk.X, padx=10, pady=(2, 8))
        ctk.CTkCheckBox(
            auto_row,
            text="Auto-calibrate before start",
            variable=self.auto_calibrate_on_start,
            font=("Segoe UI", 11),
            text_color=text_main,
            fg_color="#3f6f56",
            hover_color="#2f5a44",
            border_color="#4a5563",
        ).pack(side=tk.LEFT)

        ctk.CTkLabel(
            left_panel,
            textvariable=self.calibration_summary,
            wraplength=220,
            justify=tk.LEFT,
            text_color=text_muted,
            font=("Segoe UI", 10),
        ).pack(fill=tk.X, padx=10, pady=(0, 8))

        # Center panel: channels with horizontal scrolling.
        center_panel = ctk.CTkFrame(
            body,
            fg_color=panel_bg,
            corner_radius=14,
            border_width=1,
            border_color=panel_border,
        )
        center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        ctk.CTkLabel(
            center_panel,
            text="Channel Strips",
            font=("Segoe UI", 13, "bold"),
            text_color=text_muted,
        ).pack(anchor="w", padx=12, pady=(10, 2))

        controls_row = ctk.CTkFrame(center_panel, fg_color="transparent")
        controls_row.pack(fill=tk.X, padx=10, pady=(8, 4))

        ctk.CTkButton(
            controls_row,
            text="+ Add Model Slot",
            command=self.add_model_slot,
            font=("Segoe UI", 12, "bold"),
            width=170,
            height=36,
            corner_radius=10,
            fg_color=accent_orange,
            hover_color="#c8692f",
            text_color=text_bright,
        ).pack(side=tk.LEFT)

        ctk.CTkLabel(
            controls_row,
            text="Use horizontal scrollbar to view all channels",
            text_color=text_muted,
            font=("Segoe UI", 12),
        ).pack(side=tk.LEFT, padx=10)

        channels_canvas = tk.Canvas(center_panel, bg=panel_bg, highlightthickness=0)
        channels_canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))

        channels_h_scroll = ctk.CTkScrollbar(
            center_panel,
            orientation="horizontal",
            command=channels_canvas.xview,
            fg_color="#101319",
            button_color="#050608",
            button_hover_color="#11151b",
        )
        channels_h_scroll.pack(fill=tk.X, padx=8, pady=(0, 8))
        channels_canvas.configure(xscrollcommand=channels_h_scroll.set)

        self.slots_container = ctk.CTkFrame(channels_canvas, fg_color=panel_bg, corner_radius=0)
        self.channels_window_id = channels_canvas.create_window((0, 0), window=self.slots_container, anchor="nw")

        self.slots_container.bind(
            "<Configure>",
            lambda e: channels_canvas.configure(scrollregion=channels_canvas.bbox("all")),
        )

        def _on_channels_canvas_configure(event):
            channels_canvas.itemconfigure(self.channels_window_id, height=event.height)
            slot_height = max(260, event.height - 12)
            for slot in getattr(self, "model_slots", []):
                if hasattr(slot, "frame"):
                    slot.frame.configure(height=slot_height)

        channels_canvas.bind("<Configure>", _on_channels_canvas_configure)

        # Right panel: status and master section.
        right_panel = ctk.CTkFrame(
            body,
            fg_color=panel_bg,
            corner_radius=14,
            border_width=1,
            border_color=panel_border,
            width=300,
        )
        right_panel.pack(side=tk.LEFT, fill=tk.Y)
        right_panel.pack_propagate(False)
        ctk.CTkLabel(
            right_panel,
            text="Master / Status",
            font=("Segoe UI", 13, "bold"),
            text_color=text_muted,
        ).pack(anchor="w", padx=12, pady=(10, 2))

        ctk.CTkLabel(
            right_panel,
            text="Status Log",
            text_color=text_muted,
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w", padx=10, pady=(8, 4))

        status_wrap = ctk.CTkFrame(right_panel, fg_color="transparent")
        status_wrap.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        self.status_text = ctk.CTkTextbox(
            status_wrap,
            width=280,
            height=16,
            state="disabled",
            corner_radius=10,
            fg_color="#232830",
            text_color="#9ee9b8",
            font=("Consolas", 11),
        )
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ctk.CTkLabel(
            right_panel,
            text="SESSION TIME",
            text_color=text_muted,
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w", padx=10, pady=(0, 2))

        self.uptime_label = ctk.CTkLabel(
            right_panel,
            text="00:00:00",
            fg_color="#232830",
            text_color="#80e4ef",
            corner_radius=10,
            font=("Consolas", 20, "bold"),
            height=44,
        )
        self.uptime_label.pack(fill=tk.X, padx=10, pady=(0, 10))

        # Footer controls.
        footer = ctk.CTkFrame(desk, fg_color="transparent")
        footer.pack(fill=tk.X, padx=12, pady=(4, 12))

        footer_buttons = ctk.CTkFrame(footer, fg_color="transparent")
        footer_buttons.pack(anchor="center")

        self.start_button = ctk.CTkButton(
            footer_buttons,
            text="START STREAMING",
            command=self.start_streaming,
            font=("Segoe UI", 12, "bold"),
            height=40,
            width=220,
            corner_radius=12,
            fg_color=accent_green,
            hover_color="#459d48",
            text_color=text_bright,
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 8), pady=2)

        self.stop_button = ctk.CTkButton(
            footer_buttons,
            text="STOP STREAMING",
            command=self.stop_streaming,
            font=("Segoe UI", 12, "bold"),
            height=40,
            width=220,
            corner_radius=12,
            fg_color=accent_red,
            hover_color="#bf2a2a",
            text_color=text_bright,
            state="disabled",
        )
        self.stop_button.pack(side=tk.LEFT, padx=(8, 0), pady=2)

        self._uptime_start = None
        self._uptime_running = False
        self.add_model_slot()
        self.log_status("GUI initialized. Add model slots to begin.")

    def _tick_uptime(self):
        if not self._uptime_running:
            return
        elapsed = int(time.time() - self._uptime_start)
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        self.uptime_label.configure(text=f"{h:02d}:{m:02d}:{s:02d}")
        self.root.after(1000, self._tick_uptime)
