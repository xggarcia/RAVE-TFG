from tkinter import messagebox

from src.streaming import ModelSlot


class StreamGUISlotsMixin:
    @staticmethod
    def _slot_has_any_prior(slot):
        return (slot.prior_model is not None) or bool(slot.embedded_prior_available)

    def set_prior_usage_obj(self, slot, enabled, warn_if_missing=True):
        """Enable/disable prior safely; block enable if no compatible prior is available."""
        if enabled and not self._slot_has_any_prior(slot):
            slot.use_prior.set(False)
            if warn_if_missing:
                messagebox.showwarning(
                    "Prior Not Loaded",
                    f"Load a compatible prior in Slot {slot.slot_id + 1}, or use a model with embedded prior.",
                )
            return False

        slot.use_prior.set(bool(enabled))
        return True

    def toggle_prior_obj(self, slot):
        """Toggle prior usage for a slot from the quick UI button."""
        self.set_prior_usage_obj(slot, not slot.use_prior.get(), warn_if_missing=True)
        self.update_input_controls_obj(slot)
        self.log_status(f"Slot {slot.slot_id + 1} prior {'enabled' if slot.use_prior.get() else 'disabled'}")

    def add_model_slot(self):
        """Add a new model slot dynamically."""
        if self.is_streaming:
            messagebox.showwarning("Cannot Add Slot", "Please stop streaming before adding new slots.")
            return

        slot = ModelSlot(self.next_slot_id)
        self.model_slots.append(slot)
        self.create_model_slot_ui(slot)
        self.update_all_dropdowns()

        self.log_status(f"Added Slot {self.next_slot_id + 1}")
        self.next_slot_id += 1

    def delete_model_slot(self, slot):
        """Delete a model slot."""
        if self.is_streaming:
            messagebox.showwarning("Cannot Delete Slot", "Please stop streaming before deleting slots.")
            return

        if len(self.model_slots) <= 1:
            messagebox.showwarning("Cannot Delete", "You must have at least one slot.")
            return

        if messagebox.askyesno(
            "Delete Slot",
            f"Delete Slot {slot.slot_id + 1}?\n\nThis will remove the model and all its settings.",
        ):
            self.model_slots.remove(slot)
            if hasattr(slot, "frame"):
                slot.frame.destroy()
            self.log_status(f"Deleted Slot {slot.slot_id + 1}")

    def toggle_slot_obj(self, slot):
        if slot.is_active.get():
            if not slot.is_loaded:
                messagebox.showwarning("Model Not Loaded", f"Please load a model in Slot {slot.slot_id + 1} before activating.")
                slot.is_active.set(False)
                return
            has_any_prior = self._slot_has_any_prior(slot)
            if slot.use_prior.get() and not has_any_prior:
                messagebox.showwarning(
                    "Prior Not Loaded",
                    f"Load a compatible prior in Slot {slot.slot_id + 1}, or use a model with embedded prior.",
                )
                slot.is_active.set(False)
                return
            slot.status_var.set("Active")
            self.log_status(f"Slot {slot.slot_id + 1} activated: {slot.model_path.split('/')[-1] if slot.model_path else 'Unknown'}")
        else:
            slot.status_var.set("Loaded (Inactive)")
            self.log_status(f"Slot {slot.slot_id + 1} deactivated")

    def update_input_controls_obj(self, slot):
        has_any_prior = self._slot_has_any_prior(slot)

        # If a prior was cleared after being enabled, fail safe back to source-only mode.
        if slot.use_prior.get() and not has_any_prior:
            slot.use_prior.set(False)

        if slot.input_mode.get() == "random":
            slot.intensity_frame.pack(fill="x", pady=(4, 0))
            slot.audio_frame.pack_forget()
        else:
            slot.intensity_frame.pack_forget()
            slot.audio_frame.pack(fill="x", pady=(4, 0))

        slot.prior_frame.pack(fill="x", pady=(4, 0))
        slot.prior_load_btn.configure(state="normal")
        slot.prior_temp_scale.configure(state="normal" if slot.use_prior.get() and has_any_prior else "disabled")

        if hasattr(slot, "prior_toggle_btn"):
            slot.prior_toggle_btn.configure(
                text="Prior ON" if slot.use_prior.get() else "Prior OFF",
                fg_color="#3f6f56" if slot.use_prior.get() else "#5a626d",
                hover_color="#2f5a44" if slot.use_prior.get() else "#4b525c",
            )

        # Source temperature only affects prior path when the prior accepts source-latent channels.
        source_seed_compatible = False
        if slot.prior_model is not None:
            source_seed_compatible = slot.prior_seed_channels == slot.latent_size
        elif slot.embedded_prior_available:
            source_seed_compatible = slot.embedded_prior_seed_channels == slot.latent_size

        source_temp_enabled = (not slot.use_prior.get()) or source_seed_compatible
        if hasattr(slot, "source_temp_slider"):
            slot.source_temp_slider.configure(state="normal" if source_temp_enabled else "disabled")

        if hasattr(self, "_update_phase_ui"):
            self._update_phase_ui(slot)

    def update_input_controls(self, slot_id):
        slot = self.model_slots[slot_id]
        self.update_input_controls_obj(slot)
