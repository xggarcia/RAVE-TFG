import tkinter as tk
import customtkinter as ctk


class StreamGUIPhaseWidgetsMixin:
    """Mixin for phase interpolation UI controls within each model slot."""

    def create_phase_controls(self, slot_frame, slot):
        """Create phase interpolation controls for a slot panel."""
        phase_container = ctk.CTkFrame(slot_frame, fg_color="transparent")
        phase_container.pack(fill=tk.X, padx=8, pady=(6, 0))
        slot.phase_container = phase_container

        header_row = ctk.CTkFrame(phase_container, fg_color="transparent")
        header_row.pack(fill=tk.X)

        ctk.CTkCheckBox(
            header_row,
            text="Phase Control",
            variable=slot.phase_enabled,
            font=("Segoe UI", 10, "bold"),
            text_color="#d3dae3",
            fg_color="#3f6f56",
            hover_color="#2f5a44",
            border_color="#4a5563",
            command=lambda s=slot: self.update_input_controls_obj(s),
        ).pack(side=tk.LEFT)

        btn_row = ctk.CTkFrame(phase_container, fg_color="transparent")
        btn_row.pack(fill=tk.X, pady=(4, 0))

        ctk.CTkButton(
            btn_row,
            text="+ Anchor",
            command=lambda s=slot: self.add_phase_anchor_for_slot(s),
            width=72,
            height=26,
            corner_radius=8,
            fg_color="#49525d",
            hover_color="#3d454f",
            text_color="#d9dee5",
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT, padx=(0, 4))

        ctk.CTkButton(
            btn_row,
            text="Clear",
            command=lambda s=slot: self.clear_phase_anchors_for_slot(s),
            width=52,
            height=26,
            corner_radius=8,
            fg_color="#5a3a3a",
            hover_color="#6b4444",
            text_color="#d9dee5",
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT)

        slot.phase_anchor_label = ctk.CTkLabel(
            btn_row,
            text="No anchors",
            text_color="#9da8b5",
            font=("Segoe UI", 9, "italic"),
            anchor="w",
        )
        slot.phase_anchor_label.pack(side=tk.LEFT, padx=(8, 0))

        slot.phase_slider_frame = ctk.CTkFrame(phase_container, fg_color="transparent")

        slider_row = ctk.CTkFrame(slot.phase_slider_frame, fg_color="transparent")
        slider_row.pack(fill=tk.X, pady=(4, 0))

        ctk.CTkLabel(slider_row, text="Phase", text_color="#d3dae3", width=40).pack(side=tk.LEFT)

        slot.phase_value_label = ctk.CTkLabel(
            slider_row,
            text="0.00",
            text_color="#f2f5f8",
            font=("Segoe UI", 11, "bold"),
            width=36,
        )

        slot.phase_scale = ctk.CTkSlider(
            slider_row,
            from_=0.0,
            to=1.0,
            number_of_steps=100,
            variable=slot.phase_value,
            command=self._bind_slider_value(slot.phase_value, slot.phase_value_label, decimals=2),
            width=120,
            height=18,
            corner_radius=10,
            button_corner_radius=10,
            fg_color="#1a1f26",
            progress_color="#75839a",
            button_color="#d8dee6",
            button_hover_color="#eef2f7",
        )
        slot.phase_scale.pack(side=tk.LEFT, padx=(4, 0))
        slot.phase_value_label.pack(side=tk.LEFT, padx=(6, 0))
        slot.phase_scale.set(0.0)

    def _update_phase_ui(self, slot):
        """Refresh phase anchor label and slider visibility."""
        if not hasattr(slot, "phase_anchor_label"):
            return

        n = len(slot.phase_anchors)
        if n == 0:
            slot.phase_anchor_label.configure(text="No anchors")
        else:
            labels = " | ".join(a["label"] for a in slot.phase_anchors)
            slot.phase_anchor_label.configure(text=labels)

        if slot.phase_enabled.get() and n >= 2:
            slot.phase_slider_frame.pack(fill=tk.X, pady=(4, 0))
        else:
            slot.phase_slider_frame.pack_forget()
