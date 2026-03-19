import os
import threading
import tkinter as tk
from tkinter import messagebox

import sounddevice as sd

from src.streaming import QuickCalibrator


class StreamGUIRuntimeMixin:
    def log_status(self, message):
        if threading.current_thread() is not threading.main_thread():
            self.root.after(0, lambda msg=message: self.log_status(msg))
            return
        self.status_text.config(state="normal")
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.status_text.config(state="disabled")

    def _get_active_loaded_slots(self):
        return [slot for slot in self.model_slots if slot.is_active.get() and slot.is_loaded]

    def _validate_active_slots(self, active_slots):
        output_lengths = {slot.output_length for slot in active_slots}
        if len(output_lengths) > 1:
            lengths_msg = ", ".join(str(length) for length in sorted(output_lengths))
            messagebox.showerror(
                "Incompatible Model Outputs",
                "All active models must produce the same chunk size.\n" f"Detected output lengths: {lengths_msg}",
            )
            return False
        return True

    def _apply_calibration_result(self, result):
        self.performance_mode.set(result["recommended_mode"])
        self.sr.set(result["recommended_sr"])

    def run_quick_calibration(self):
        slots = self.calibrator.collect_slots(self.model_slots)
        if not slots:
            messagebox.showwarning(
                "Calibration Unavailable",
                "Load at least one model first. Calibration needs model decode timing.",
            )
            return

        signature = self.calibrator.build_signature(slots, self.sr.get())

        self.calibrate_button.config(state="disabled")
        try:
            result = self.calibrator.run(slots, self.sr.get())
        except Exception as e:
            messagebox.showerror("Calibration Error", str(e))
            self.log_status(f"[CAL] Failed: {e}")
            return
        finally:
            self.calibrate_button.config(state="normal")

        self.last_calibration_signature = signature
        self.last_calibration_result = result
        self.calibration_summary.set(QuickCalibrator.format_summary(result))

        self.log_status(self.calibration_summary.get())
        apply_now = messagebox.askyesno(
            "Calibration Complete",
            "Quick calibration finished.\n\n"
            f"Recommended mode: {result['recommended_mode']}\n"
            f"Recommended sample rate: {result['recommended_sr']} Hz\n"
            f"Estimated safe slots at current quality: ~{result['safe_slots']}\n\n"
            "Apply these recommendations now?",
        )
        if apply_now:
            self._apply_calibration_result(result)
            self.log_status("[CAL] Recommendations applied.")

    def _ensure_calibrated_for_slots(self, active_slots):
        if not self.auto_calibrate_on_start.get():
            return True

        signature = self.calibrator.build_signature(active_slots, self.sr.get())
        if self.last_calibration_signature == signature and self.last_calibration_result is not None:
            return True

        self.calibrate_button.config(state="disabled")
        try:
            result = self.calibrator.run(active_slots, self.sr.get())
        except Exception as e:
            messagebox.showerror("Auto Calibration Error", str(e))
            self.log_status(f"[CAL] Auto calibration failed: {e}")
            return False
        finally:
            self.calibrate_button.config(state="normal")

        self.last_calibration_signature = signature
        self.last_calibration_result = result
        self.calibration_summary.set(QuickCalibrator.format_summary(result))
        self.log_status(self.calibration_summary.get())

        self._apply_calibration_result(result)
        self.log_status("[CAL] Auto-calibration recommendations applied before streaming.")
        return True

    def start_streaming(self):
        active_slots = self._get_active_loaded_slots()

        if not active_slots:
            messagebox.showwarning(
                "No Active Models",
                "Please load and activate at least one model before starting.",
            )
            return

        if not self._validate_active_slots(active_slots):
            return

        if not self._ensure_calibrated_for_slots(active_slots):
            return

        self.chunk_samples = active_slots[0].output_length
        actual_duration = self.chunk_samples / self.sr.get()

        self.log_status("=" * 50)
        self.log_status(f"Starting streaming with {len(active_slots)} active model(s)")
        self.log_status(f"Sample Rate: {self.sr.get()} Hz")
        self.log_status(f"Chunk: {self.chunk_samples} samples ({actual_duration:.3f}s)")

        for slot in active_slots:
            input_info = "Random" if slot.input_mode.get() == "random" else f"Audio: {os.path.basename(slot.audio_file_path) if slot.audio_file_path else 'None'}"
            self.log_status(f"  Slot {slot.slot_id + 1}: {os.path.basename(slot.model_path)} [{input_info}]")
            self.log_status(f"    Gain={slot.gain.get():.2f}, Temp={slot.temperature.get():.2f}, Smooth={slot.smoothing.get():.2f}")

        try:
            self.stream = sd.OutputStream(
                samplerate=self.sr.get(),
                channels=1,
                dtype="float32",
                blocksize=self.chunk_samples,
            )
            self.stream.start()
            self.log_status("Audio stream initialized")
        except Exception as e:
            messagebox.showerror("Audio Error", f"Failed to initialize audio:\n{str(e)}")
            return

        self.is_streaming = True
        self.engine.configure(
            stream=self.stream,
            sr=self.sr.get(),
            chunk_samples=self.chunk_samples,
            performance_mode=self.performance_mode.get(),
        )
        self.engine.start()
        self.stream_thread = self.engine.consumer_thread

        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")

        self.log_status(
            f"Mode: {self.performance_mode.get()} | Queue: {self.engine.queue_maxsize} | "
            f"Base decode stride: {self.engine.base_decode_stride}"
        )
        self.log_status("STREAMING STARTED - Mixing active models in real-time")

    def stop_streaming(self):
        self.is_streaming = False
        self.engine.stop()
        self.stream_thread = None

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.log_status("STREAMING STOPPED")
