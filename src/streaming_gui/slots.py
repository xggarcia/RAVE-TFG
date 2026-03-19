from tkinter import messagebox

from src.streaming import ModelSlot


class StreamGUISlotsMixin:
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
            slot.status_var.set("Active")
            self.log_status(f"Slot {slot.slot_id + 1} activated: {slot.model_path.split('/')[-1] if slot.model_path else 'Unknown'}")
        else:
            slot.status_var.set("Loaded (Inactive)")
            self.log_status(f"Slot {slot.slot_id + 1} deactivated")

    def update_input_controls_obj(self, slot):
        if slot.input_mode.get() == "random":
            slot.intensity_frame.pack(side="left", padx=10)
            slot.audio_frame.pack_forget()
        else:
            slot.intensity_frame.pack_forget()
            slot.audio_frame.pack(side="left", padx=10)

    def update_input_controls(self, slot_id):
        slot = self.model_slots[slot_id]
        self.update_input_controls_obj(slot)
