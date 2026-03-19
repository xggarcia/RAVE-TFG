import os
from tkinter import filedialog, messagebox

import librosa
import torch


class StreamGUIModelIOMixin:
    def select_audio_for_slot_obj(self, slot):
        filename = filedialog.askopenfilename(
            title=f"Select Audio File for Slot {slot.slot_id + 1}",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg"), ("WAV Files", "*.wav"), ("All Files", "*.*")],
            initialdir=os.path.join(os.getcwd(), "input_data"),
        )

        if not filename:
            return

        if not slot.is_loaded:
            messagebox.showwarning("Model Not Loaded", "Please load a model first before selecting audio input.")
            return

        try:
            self.log_status(f"Slot {slot.slot_id + 1}: Loading audio file {os.path.basename(filename)}...")
            audio, _ = librosa.load(filename, sr=self.sr.get(), mono=True)
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                encoded = slot.model.encode(audio_tensor)

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
        slot = self.model_slots[slot_id]
        self.select_audio_for_slot_obj(slot)

    def load_model_to_slot_obj(self, slot):
        model_path = slot.model_var.get()

        if model_path == "[No Model Loaded]" or not model_path:
            messagebox.showwarning("No Model", "Please select a model from the dropdown.")
            return

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
            model = torch.jit.load(full_path).eval()

            with torch.no_grad():
                latent_size = None
                for test_channels in [1, 2, 8, 16, 32, 64, 128]:
                    try:
                        test_z = torch.randn(1, test_channels, 128)
                        _ = model.decode(test_z)
                        latent_size = test_channels
                        self.log_status(f"  Detected latent channels: {latent_size}")
                        break
                    except Exception:
                        continue

                if latent_size is None:
                    raise Exception("Could not determine latent channel size")

                latent_length = 1
                output_length = None
                max_latent = 16384

                while latent_length <= max_latent:
                    try:
                        test_z = torch.randn(1, latent_size, latent_length)
                        test_output = model.decode(test_z)
                        output_length = test_output.shape[-1]
                        self.log_status(
                            f"  Detected: latent_size={latent_size}, latent_length={latent_length}, output={output_length}"
                        )
                        break
                    except Exception:
                        latent_length *= 2

                if output_length is None:
                    raise Exception(f"Could not determine model dimensions (tested up to {max_latent})")

            slot.model = model
            slot.model_path = full_path
            slot.latent_size = latent_size
            slot.latent_length = latent_length
            slot.output_length = output_length
            slot.prev_z = None
            slot.is_loaded = True
            slot.status_var.set("Loaded (Inactive)")

            self.log_status(f"Slot {slot.slot_id + 1}: Model loaded successfully")
            self.log_status(f"  Latent: {slot.latent_size}x{slot.latent_length}, Output: {slot.output_length} samples")

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load model:\n{str(e)}")
            self.log_status(f"Slot {slot.slot_id + 1}: Load failed - {str(e)}")

    def load_model_to_slot(self, slot_id):
        slot = self.model_slots[slot_id]
        self.load_model_to_slot_obj(slot)

    def browse_model_for_slot_obj(self, slot):
        filename = filedialog.askopenfilename(
            title=f"Select Model for Slot {slot.slot_id + 1}",
            filetypes=[("TorchScript Models", "*.ts"), ("All Files", "*.*")],
            initialdir=os.path.join(os.getcwd(), "models"),
        )

        if filename:
            slot.model_var.set(os.path.basename(filename))
            if filename not in self.available_models:
                self.available_models.append(filename)
                self.update_all_dropdowns()

    def browse_model_for_slot(self, slot_id):
        slot = self.model_slots[slot_id]
        self.browse_model_for_slot_obj(slot)

    def update_all_dropdowns(self):
        model_names = ["[No Model Loaded]"] + [os.path.basename(m) for m in self.available_models]
        for slot in self.model_slots:
            if hasattr(slot, "combo"):
                slot.combo["values"] = model_names

    def discover_models(self):
        search_dirs = [
            os.path.join(os.getcwd(), "models", "demo_model"),
            os.path.join(os.getcwd(), "models", "user_model", "exported_model"),
        ]

        self.available_models = []
        for directory in search_dirs:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    if file.endswith(".ts"):
                        full_path = os.path.join(directory, file)
                        self.available_models.append(full_path)

        self.update_all_dropdowns()
        self.log_status(f"Found {len(self.available_models)} available models")
