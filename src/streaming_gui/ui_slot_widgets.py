import tkinter as tk
import customtkinter as ctk


class StreamGUISlotWidgetsMixin:
    @staticmethod
    def _bind_slider_value(variable, value_label, decimals=2):
        def _on_change(value):
            variable.set(float(value))
            value_label.configure(text=f"{float(value):.{decimals}f}")

        return _on_change

    def create_model_slot_ui(self, slot):
        """Create a complete model slot with all controls."""
        slot_id = slot.slot_id
        panel_bg = "#2b3038"
        colors = ["#84a9cf", "#c89f97", "#cbb784", "#b3a2c9", "#7fb9b2", "#c6a687", "#9eacba", "#8ca8c5"]
        color_index = slot_id % len(colors)

        slot_frame = ctk.CTkFrame(
            self.slots_container,
            fg_color=panel_bg,
            corner_radius=14,
            border_width=1,
            border_color="#12161d",
        )
        slot_frame.pack(side=tk.LEFT, fill=tk.Y, pady=6, padx=6)
        slot_frame.configure(width=230, height=620)
        slot_frame.pack_propagate(False)
        slot.frame = slot_frame

        title_row = ctk.CTkFrame(slot_frame, fg_color="transparent")
        title_row.pack(fill=tk.X, padx=8, pady=(8, 2))
        ctk.CTkLabel(
            title_row,
            text=f"Model Slot {slot_id + 1}",
            font=("Segoe UI", 12, "bold"),
            text_color=colors[color_index],
        ).pack(side=tk.LEFT)
        delete_btn = ctk.CTkButton(
            title_row,
            text="🗑",
            command=lambda s=slot: self.delete_model_slot(s),
            fg_color="#e74c3c",
            hover_color="#cc3a2c",
            text_color="#d9dee5",
            font=("Segoe UI", 10, "bold"),
            width=32,
            height=26,
            corner_radius=8,
        )
        delete_btn.pack(side=tk.RIGHT)

        top_row = ctk.CTkFrame(slot_frame, fg_color="transparent")
        top_row.pack(fill=tk.X, padx=8, pady=(6, 4))

        active_check = ctk.CTkCheckBox(
            top_row,
            text="ACTIVE",
            variable=slot.is_active,
            font=("Segoe UI", 11, "bold"),
            text_color=colors[color_index],
            fg_color="#3f6f56",
            hover_color="#2f5a44",
            border_color="#4a5563",
            command=lambda s=slot: self.toggle_slot_obj(s),
        )
        active_check.pack(side=tk.LEFT, padx=(0, 6))

        status_label = ctk.CTkLabel(
            top_row,
            textvariable=slot.status_var,
            font=("Segoe UI", 10, "italic"),
            text_color="#a7b0bc",
            anchor="e",
        )
        status_label.pack(side=tk.RIGHT, padx=(6, 0))

        model_row_top = ctk.CTkFrame(slot_frame, fg_color="transparent")
        model_row_top.pack(fill=tk.X, padx=8, pady=(2, 2))

        model_combo = ctk.CTkOptionMenu(
            model_row_top,
            values=[slot.model_var.get()],
            variable=slot.model_var,
            width=150,
            dynamic_resizing=False,
            corner_radius=8,
            fg_color="#3a424d",
            button_color="#2a3038",
            button_hover_color="#232830",
            dropdown_fg_color="#2a3038",
            text_color="#d3dae3",
        )
        model_combo.pack(side=tk.LEFT, padx=(0, 4))
        slot.combo = model_combo

        load_btn = ctk.CTkButton(
            model_row_top,
            text="Load",
            command=lambda s=slot: self.load_model_to_slot_obj(s),
            font=("Segoe UI", 9, "bold"),
            width=52,
            height=30,
            corner_radius=8,
            fg_color=colors[color_index],
            hover_color="#556170",
            text_color="#d9dee5",
        )
        load_btn.pack(side=tk.LEFT, padx=2)

        model_row_bottom = ctk.CTkFrame(slot_frame, fg_color="transparent")
        model_row_bottom.pack(fill=tk.X, padx=8, pady=(0, 4))

        browse_btn = ctk.CTkButton(
            model_row_bottom,
            text="Browse...",
            command=lambda s=slot: self.browse_model_for_slot_obj(s),
            width=64,
            height=28,
            corner_radius=8,
            fg_color="#49525d",
            hover_color="#3d454f",
            text_color="#d9dee5",
        )
        browse_btn.pack(side=tk.LEFT, padx=2)

        prior_btn = ctk.CTkButton(
            model_row_bottom,
            text="Prior...",
            command=lambda s=slot: self.load_prior_for_slot_obj(s),
            width=64,
            height=28,
            corner_radius=8,
            fg_color="#49525d",
            hover_color="#3d454f",
            text_color="#d9dee5",
        )
        prior_btn.pack(side=tk.LEFT, padx=2)

        slot.prior_toggle_btn = ctk.CTkButton(
            model_row_bottom,
            text="Prior OFF",
            command=lambda s=slot: self.toggle_prior_obj(s),
            width=74,
            height=28,
            corner_radius=8,
            fg_color="#5a626d",
            hover_color="#4b525c",
            text_color="#d9dee5",
            font=("Segoe UI", 8, "bold"),
        )
        slot.prior_toggle_btn.pack(side=tk.LEFT, padx=2)

        input_row = ctk.CTkFrame(slot_frame, fg_color="transparent")
        input_row.pack(fill=tk.X, padx=4, pady=(5, 0))

        ctk.CTkLabel(input_row, text="Input:", text_color="#d3dae3", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=(0, 3))

        ctk.CTkRadioButton(
            input_row,
            text="Random",
            variable=slot.input_mode,
            value="random",
            text_color="#d3dae3",
            fg_color="#3f6f56",
            hover_color="#2f5a44",
            command=lambda s=slot: self.update_input_controls_obj(s),
        ).pack(side=tk.LEFT, padx=3)

        ctk.CTkRadioButton(
            input_row,
            text="Audio",
            variable=slot.input_mode,
            value="audio",
            text_color="#d3dae3",
            fg_color="#3f6f56",
            hover_color="#2f5a44",
            command=lambda s=slot: self.update_input_controls_obj(s),
        ).pack(side=tk.LEFT, padx=3)

        ctk.CTkCheckBox(
            input_row,
            text="Use Prior",
            variable=slot.use_prior,
            text_color="#d3dae3",
            fg_color="#3f6f56",
            hover_color="#2f5a44",
            border_color="#4a5563",
            command=lambda s=slot: (self.set_prior_usage_obj(s, s.use_prior.get(), warn_if_missing=True), self.update_input_controls_obj(s)),
        ).pack(side=tk.LEFT, padx=4)

        input_detail_row = ctk.CTkFrame(slot_frame, fg_color="transparent")
        input_detail_row.pack(fill=tk.X, padx=10, pady=(2, 0))

        slot.intensity_frame = ctk.CTkFrame(input_detail_row, fg_color="transparent")
        intensity_label_col = ctk.CTkFrame(slot.intensity_frame, fg_color="transparent")
        intensity_label_col.pack(side=tk.LEFT, padx=(0, 8))
        ctk.CTkLabel(intensity_label_col, text="Intensity", text_color="#d3dae3", width=64).pack(side=tk.TOP)
        intensity_slider_col = ctk.CTkFrame(slot.intensity_frame, fg_color="transparent")
        intensity_slider_col.pack(side=tk.LEFT)
        slot.intensity_value_label = ctk.CTkLabel(
            intensity_slider_col,
            text=f"{slot.random_intensity.get():.1f}",
            text_color="#f2f5f8",
            font=("Segoe UI", 12, "bold"),
        )
        slot.intensity_scale = ctk.CTkSlider(
            slot.intensity_frame,
            from_=0.1,
            to=3.0,
            number_of_steps=29,
            variable=slot.random_intensity,
            command=self._bind_slider_value(slot.random_intensity, slot.intensity_value_label, decimals=1),
            width=130,
            height=18,
            corner_radius=10,
            button_corner_radius=10,
            fg_color="#1a1f26",
            progress_color="#75839a",
            button_color="#d8dee6",
            button_hover_color="#eef2f7",
        )
        slot.intensity_scale.pack(in_=intensity_slider_col, side=tk.TOP)
        slot.intensity_value_label.pack(side=tk.TOP, pady=(3, 0))
        slot.intensity_scale.set(slot.random_intensity.get())
        slot.intensity_frame.pack(fill=tk.X, pady=(4, 0))

        slot.audio_frame = ctk.CTkFrame(input_detail_row, fg_color="transparent")
        ctk.CTkButton(
            slot.audio_frame,
            text="Select Audio",
            command=lambda s=slot: self.select_audio_for_slot_obj(s),
            width=96,
            height=28,
            corner_radius=8,
            fg_color="#49525d",
            hover_color="#3d454f",
            text_color="#d9dee5",
        ).pack(
            side=tk.LEFT, padx=5
        )
        ctk.CTkCheckBox(
            slot.audio_frame,
            text="Loop",
            variable=slot.loop_audio,
            text_color="#d3dae3",
            fg_color="#3f6f56",
            hover_color="#2f5a44",
            border_color="#4a5563",
        ).pack(side=tk.LEFT)
        slot.audio_frame.pack(fill=tk.X, pady=(4, 0))

        slot.prior_frame = ctk.CTkFrame(input_detail_row, fg_color="transparent")
        slot.prior_load_btn = ctk.CTkButton(
            slot.prior_frame,
            text="Load Prior",
            command=lambda s=slot: self.load_prior_for_slot_obj(s),
            width=96,
            height=28,
            corner_radius=8,
            fg_color="#49525d",
            hover_color="#3d454f",
            text_color="#d9dee5",
        )
        slot.prior_load_btn.pack(side=tk.LEFT, padx=5)
        ctk.CTkLabel(
            slot.prior_frame,
            textvariable=slot.prior_status_var,
            text_color="#9da8b5",
            width=24,
            anchor="w",
        ).pack(side=tk.LEFT, padx=5)
        prior_temp_col = ctk.CTkFrame(slot.prior_frame, fg_color="transparent")
        prior_temp_col.pack(side=tk.LEFT, padx=(4, 0))
        ctk.CTkLabel(prior_temp_col, text="Prior Temp", text_color="#d3dae3", width=68).pack(side=tk.TOP)
        slot.prior_temp_value_label = ctk.CTkLabel(
            prior_temp_col,
            text=f"{slot.prior_temperature.get():.1f}",
            text_color="#f2f5f8",
            font=("Segoe UI", 12, "bold"),
        )
        slot.prior_temp_scale = ctk.CTkSlider(
            slot.prior_frame,
            from_=0.1,
            to=3.0,
            number_of_steps=29,
            variable=slot.prior_temperature,
            command=self._bind_slider_value(slot.prior_temperature, slot.prior_temp_value_label, decimals=1),
            width=120,
            height=18,
            corner_radius=10,
            button_corner_radius=10,
            fg_color="#1a1f26",
            progress_color="#75839a",
            button_color="#d8dee6",
            button_hover_color="#eef2f7",
        )
        slot.prior_temp_scale.pack(in_=prior_temp_col, side=tk.TOP)
        slot.prior_temp_value_label.pack(side=tk.TOP, pady=(3, 0))
        slot.prior_temp_scale.set(slot.prior_temperature.get())
        slot.prior_frame.pack(fill=tk.X, pady=(4, 0))

        self.update_input_controls_obj(slot)

        params_row = ctk.CTkFrame(slot_frame, fg_color="transparent")
        params_row.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 0))

        slot.gain_slider = self.create_slot_slider(params_row, "Gain", slot.gain, 0.0, 1.0, 0)
        slot.source_temp_slider = self.create_slot_slider(params_row, "Src Temp", slot.temperature, 0.1, 3.0, 1)
        slot.smooth_slider = self.create_slot_slider(params_row, "Smooth", slot.smoothing, 0.0, 0.95, 2)

    @staticmethod
    def create_slot_slider(parent, label, variable, min_val, max_val, column):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=0, column=column, padx=8, pady=0, sticky="nsew")

        ctk.CTkLabel(frame, text=label, text_color="#d3dae3", width=64, anchor="center").pack(side=tk.TOP, pady=(0, 6))

        value_label = ctk.CTkLabel(
            frame,
            text=f"{variable.get():.2f}",
            text_color="#f2f5f8",
            font=("Segoe UI", 13, "bold"),
            width=64,
            anchor="center",
        )

        steps = max(1, int(round((max_val - min_val) / 0.01)))
        slider = ctk.CTkSlider(
            frame,
            from_=min_val,
            to=max_val,
            number_of_steps=steps,
            orientation="vertical",
            variable=variable,
            command=StreamGUISlotWidgetsMixin._bind_slider_value(variable, value_label, decimals=2),
            width=18,
            height=300,
            corner_radius=10,
            button_corner_radius=10,
            fg_color="#1a1f26",
            progress_color="#75839a",
            button_color="#d8dee6",
            button_hover_color="#eef2f7",
        )
        slider.pack(side=tk.TOP, fill=tk.Y, expand=True)
        value_label.pack(side=tk.BOTTOM, pady=(5, 0))
        slider.set(variable.get())
        parent.grid_columnconfigure(column, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        return slider
