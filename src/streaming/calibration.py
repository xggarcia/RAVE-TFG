import statistics
import time

import torch


class QuickCalibrator:
    """Runs short decode benchmarks and generates runtime recommendations."""

    def __init__(self, logger, validate_slots):
        self.log = logger
        self.validate_slots = validate_slots

    def collect_slots(self, model_slots):
        active_slots = [slot for slot in model_slots if slot.is_active.get() and slot.is_loaded]
        if active_slots:
            return active_slots
        return [slot for slot in model_slots if slot.is_loaded]

    def build_signature(self, slots, sr):
        slot_key = tuple(
            (slot.model_path, slot.latent_size, slot.latent_length, slot.output_length)
            for slot in sorted(slots, key=lambda s: s.slot_id)
        )
        return (sr, slot_key)

    def recommend_settings(self, avg_slot_decode_ms, chunk_budget_ms, desired_slots, current_sr):
        if avg_slot_decode_ms <= 0.0:
            return {
                "recommended_mode": "Balanced",
                "recommended_sr": current_sr,
                "safe_slots": desired_slots,
                "estimated_load": 0.0,
            }

        estimated_load = (avg_slot_decode_ms * desired_slots) / chunk_budget_ms
        safe_slots = max(1, int((chunk_budget_ms * 0.85) / avg_slot_decode_ms))

        if estimated_load <= 0.55:
            mode = "Quality"
        elif estimated_load <= 0.90:
            mode = "Balanced"
        else:
            mode = "Max Stability"

        recommended_sr = current_sr
        if desired_slots > safe_slots and current_sr > 22050:
            recommended_sr = 22050

        return {
            "recommended_mode": mode,
            "recommended_sr": recommended_sr,
            "safe_slots": safe_slots,
            "estimated_load": estimated_load,
        }

    def run(self, slots, sr):
        if not slots:
            raise ValueError("No loaded models available for calibration.")

        if not self.validate_slots(slots):
            raise ValueError("Calibration aborted due to incompatible model outputs.")

        chunk_samples = slots[0].output_length
        chunk_budget_ms = (chunk_samples / sr) * 1000.0

        warmup_iters = 2
        measure_iters = 8
        per_slot_decode_ms = []

        self.log("[CAL] Running quick benchmark...")
        for i in range(warmup_iters + measure_iters):
            for slot in slots:
                start = time.perf_counter()
                with torch.no_grad():
                    z = torch.randn(1, slot.latent_size, slot.latent_length)
                    _ = slot.model.decode(z)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                if i >= warmup_iters:
                    per_slot_decode_ms.append(elapsed_ms)

        avg_slot_decode_ms = statistics.fmean(per_slot_decode_ms) if per_slot_decode_ms else 0.0
        desired_slots = len(slots)
        recommendation = self.recommend_settings(
            avg_slot_decode_ms=avg_slot_decode_ms,
            chunk_budget_ms=chunk_budget_ms,
            desired_slots=desired_slots,
            current_sr=sr,
        )

        return {
            "avg_slot_decode_ms": avg_slot_decode_ms,
            "chunk_budget_ms": chunk_budget_ms,
            "desired_slots": desired_slots,
            "safe_slots": recommendation["safe_slots"],
            "estimated_load": recommendation["estimated_load"],
            "recommended_mode": recommendation["recommended_mode"],
            "recommended_sr": recommendation["recommended_sr"],
        }

    @staticmethod
    def format_summary(result):
        return (
            "Calibration: "
            f"slot_decode={result['avg_slot_decode_ms']:.1f}ms | "
            f"budget={result['chunk_budget_ms']:.1f}ms | "
            f"safe_slots~{result['safe_slots']} | "
            f"rec={result['recommended_mode']} @ {result['recommended_sr']}Hz"
        )
