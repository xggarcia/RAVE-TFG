"""Input-section and prior-controls builder mixin for StreamGUI slot panels."""
import tkinter as tk
import customtkinter as ctk


class _SlotInputsMixin:
    def _build_input_controls(self, slot_frame, slot):
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

        intensity_row = ctk.CTkFrame(slot.intensity_frame, fg_color="transparent")
        intensity_row.pack(fill=tk.X, pady=(0, 4))
        ctk.CTkLabel(intensity_row, text="Intensity", text_color="#d3dae3", width=64).pack(side=tk.LEFT, padx=(0, 8))
        slot.intensity_value_label = ctk.CTkLabel(
            intensity_row,
            text=f"{slot.random_intensity.get():.1f}",
            text_color="#f2f5f8",
            font=("Segoe UI", 12, "bold"),
            width=36,
        )
        slot.intensity_scale = ctk.CTkSlider(
            intensity_row,
            from_=0.1,
            to=3.0,
            number_of_steps=29,
            variable=slot.random_intensity,
            command=self._bind_slider_value(slot.random_intensity, slot.intensity_value_label, decimals=1),
            width=100,
            height=18,
            corner_radius=10,
            button_corner_radius=10,
            fg_color="#1a1f26",
            progress_color="#75839a",
            button_color="#d8dee6",
            button_hover_color="#eef2f7",
        )
        slot.intensity_scale.pack(side=tk.LEFT)
        slot.intensity_value_label.pack(side=tk.LEFT, padx=(6, 0))
        slot.intensity_scale.set(slot.random_intensity.get())

        density_row = ctk.CTkFrame(slot.intensity_frame, fg_color="transparent")
        density_row.pack(fill=tk.X, pady=(0, 2))
        ctk.CTkLabel(density_row, text="Density", text_color="#d3dae3", width=64).pack(side=tk.LEFT, padx=(0, 8))
        slot.density_value_label = ctk.CTkLabel(
            density_row,
            text=f"{slot.density.get():.2f}",
            text_color="#f2f5f8",
            font=("Segoe UI", 12, "bold"),
            width=36,
        )
        slot.density_scale = ctk.CTkSlider(
            density_row,
            from_=0.05,
            to=1.0,
            number_of_steps=19,
            variable=slot.density,
            command=self._bind_slider_value(slot.density, slot.density_value_label, decimals=2),
            width=100,
            height=18,
            corner_radius=10,
            button_corner_radius=10,
            fg_color="#1a1f26",
            progress_color="#75839a",
            button_color="#d8dee6",
            button_hover_color="#eef2f7",
        )
        slot.density_scale.pack(side=tk.LEFT)
        slot.density_value_label.pack(side=tk.LEFT, padx=(6, 0))
        slot.density_scale.set(slot.density.get())

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
        ).pack(side=tk.LEFT, padx=5)
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
        self.create_phase_controls(slot_frame, slot)
