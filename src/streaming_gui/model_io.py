import os
import re
from tkinter import filedialog, messagebox, simpledialog

import librosa
import torch

from src.streaming.phase_control import encode_phase_anchor, load_phase_anchors


class StreamGUIModelIOMixin:
    @staticmethod
    def _coerce_prior_latent(z, latent_size, latent_length):
        if isinstance(z, (tuple, list)):
            tensors = [item for item in z if torch.is_tensor(item)]
            if not tensors:
                raise ValueError("Prior output tuple/list has no tensor")
            z = tensors[0]

        if not torch.is_tensor(z):
            raise ValueError("Prior output is not a tensor")

        if z.dim() == 2:
            z = z.unsqueeze(0)
        elif z.dim() != 3:
            raise ValueError(f"Unsupported prior latent rank: {z.dim()}")

        if z.shape[1] != latent_size and z.shape[2] == latent_size:
            z = z.transpose(1, 2)

        if z.shape[1] != latent_size:
            raise ValueError(f"Prior latent channels mismatch: got {z.shape[1]}, expected {latent_size}")

        if z.shape[2] > latent_length:
            z = z[:, :, :latent_length]
        elif z.shape[2] < latent_length:
            z = torch.nn.functional.pad(z, (0, latent_length - z.shape[2]))

        return z

    def _find_compatible_prior_seed(self, prior_model, decoder_model, latent_size, latent_length):
        with torch.no_grad():
            for seed_channels in (1, 1024):
                try:
                    seed = torch.zeros(1, seed_channels, latent_length).float()
                    z_out = prior_model.prior(seed)
                    z = self._coerce_prior_latent(z_out, latent_size, latent_length)
                    _ = decoder_model.decode(z)
                    return seed_channels
                except Exception:
                    continue
        return None

    def load_prior_for_slot_obj(self, slot):
        if not slot.is_loaded or slot.model is None:
            messagebox.showwarning("Decoder Required", "Load the decoder model in this slot before loading a prior.")
            return

        filename = filedialog.askopenfilename(
            title=f"Select Prior Model for Slot {slot.slot_id + 1}",
            filetypes=[("TorchScript Models", "*.ts"), ("All Files", "*.*")],
            initialdir=os.path.join(os.getcwd(), "models"),
        )

        if not filename:
            return

        try:
            self.log_status(f"Slot {slot.slot_id + 1}: Loading prior {os.path.basename(filename)}...")
            prior_model = torch.jit.load(filename).eval()

            if not hasattr(prior_model, "prior"):
                raise Exception("Selected file does not expose a prior(...) method")

            with torch.no_grad():
                compatible_seed = self._find_compatible_prior_seed(
                    prior_model=prior_model,
                    decoder_model=slot.model,
                    latent_size=slot.latent_size,
                    latent_length=slot.latent_length,
                )

            if compatible_seed is None:
                raise Exception("Could not find a compatible prior seed layout for this decoder")

            slot.prior_model = prior_model
            slot.prior_model_path = filename
            slot.prior_seed_channels = compatible_seed
            slot.prior_status_var.set(os.path.basename(filename))
            slot.prior_needs_warmup = True
            slot.prior_chunks_generated = 0
            self.log_status(
                f"Slot {slot.slot_id + 1}: Prior loaded successfully (seed channels: {compatible_seed})"
            )
            self.update_input_controls_obj(slot)

        except Exception as e:
            messagebox.showerror("Prior Load Error", f"Failed to load compatible prior:\n{str(e)}")
            self.log_status(f"Slot {slot.slot_id + 1}: Prior load failed - {str(e)}")

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
            slot.model_sr = None

            # Common export naming convention includes sample rate, e.g. *_r48000_*.ts
            name_match = re.search(r"_r(\d+)_", os.path.basename(full_path))
            if name_match:
                try:
                    slot.model_sr = int(name_match.group(1))
                except ValueError:
                    slot.model_sr = None

            slot.latent_size = latent_size
            slot.latent_length = latent_length
            slot.output_length = output_length
            slot.prev_z = None
            slot.is_loaded = True
            slot.status_var.set("Loaded (Inactive)")

            # Decoder changed; clear prior state until user loads a compatible prior.
            slot.prior_model = None
            slot.prior_model_path = None
            slot.prior_seed_channels = None
            slot.embedded_prior_available = False
            slot.embedded_prior_seed_channels = None
            slot.prior_needs_warmup = True
            slot.prior_chunks_generated = 0

            if hasattr(model, "prior"):
                embedded_seed = self._find_compatible_prior_seed(
                    prior_model=model,
                    decoder_model=model,
                    latent_size=slot.latent_size,
                    latent_length=slot.latent_length,
                )
                if embedded_seed is not None:
                    slot.embedded_prior_available = True
                    slot.embedded_prior_seed_channels = embedded_seed
                    slot.prior_status_var.set(f"Embedded Prior (seed {embedded_seed})")
                else:
                    slot.prior_status_var.set("No Prior Loaded")
            else:
                slot.prior_status_var.set("No Prior Loaded")

            # Auto-detect phase anchors saved alongside the model
            slot.phase_anchors.clear()
            slot.phase_enabled.set(False)
            slot.phase_value.set(0.0)
            phases_path = os.path.splitext(full_path)[0] + "_phases.json"
            if os.path.isfile(phases_path):
                try:
                    slot.phase_anchors = load_phase_anchors(phases_path)
                    labels = " | ".join(a["label"] for a in slot.phase_anchors)
                    self.log_status(f"  Phase anchors auto-loaded: {labels}")
                except Exception as phase_err:
                    self.log_status(f"  Phase anchors file found but failed to load: {phase_err}")

            self.log_status(f"Slot {slot.slot_id + 1}: Model loaded successfully")
            self.log_status(f"  Latent: {slot.latent_size}x{slot.latent_length}, Output: {slot.output_length} samples")
            if slot.model_sr is not None:
                self.log_status(f"  Detected model sample rate: {slot.model_sr} Hz")
            self.update_input_controls_obj(slot)

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
                try:
                    slot.combo.configure(values=model_names)
                except Exception:
                    slot.combo["values"] = model_names

            current = slot.model_var.get()
            if current not in model_names:
                slot.model_var.set("[No Model Loaded]")

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

    def add_phase_anchor_for_slot(self, slot):
        if not slot.is_loaded or slot.model is None:
            messagebox.showwarning("Model Required", "Load a model before adding phase anchors.")
            return

        filename = filedialog.askopenfilename(
            title=f"Select Phase Reference Audio for Slot {slot.slot_id + 1}",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg"), ("WAV Files", "*.wav"), ("All Files", "*.*")],
            initialdir=os.path.join(os.getcwd(), "input_data"),
        )

        if not filename:
            return

        label = simpledialog.askstring(
            "Phase Label",
            f"Name for this phase anchor (e.g. 'Soft Rain', 'Storm'):",
            parent=self.root,
        )

        if not label:
            label = f"Phase {len(slot.phase_anchors) + 1}"

        try:
            self.log_status(f"Slot {slot.slot_id + 1}: Encoding phase anchor '{label}'...")
            anchor = encode_phase_anchor(slot.model, filename, self.sr.get())
            anchor["label"] = label
            slot.phase_anchors.append(anchor)

            labels = " | ".join(a["label"] for a in slot.phase_anchors)
            self.log_status(f"Slot {slot.slot_id + 1}: Phase anchors [{labels}]")
            self.update_input_controls_obj(slot)

        except Exception as e:
            messagebox.showerror("Phase Anchor Error", f"Failed to encode phase anchor:\n{str(e)}")
            self.log_status(f"Slot {slot.slot_id + 1}: Phase anchor failed - {str(e)}")

    def clear_phase_anchors_for_slot(self, slot):
        slot.phase_anchors.clear()
        slot.phase_value.set(0.0)
        slot.phase_enabled.set(False)
        self.log_status(f"Slot {slot.slot_id + 1}: Phase anchors cleared")
        self.update_input_controls_obj(slot)
