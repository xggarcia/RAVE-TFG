import os
import traceback


def run_generate_audio(ctx):
    print("\n--- Generate Audio from Model ---")
    mode = ctx.model_mode_prompt()
    if mode == "9":
        return

    if mode == "1":
        model_path = ctx.pick_model_path(default_demo=True)
        if not model_path:
            ctx.pause()
            return

        audio_path = "input_data/demo_data/audio1.wav"
        if model_path != "models/demo_model/demo_model.ts":
            audio_path = input("Reference audio path (required for user model): ").strip()
            if not audio_path or not os.path.exists(audio_path):
                print("[X] Invalid audio path.")
                ctx.pause()
                return

        output_name = input("Output name [generated]: ").strip() or "generated"
        ctx.generate_fn(model_path=model_path, audio_path=audio_path, output_name=output_name, duration=30)
        ctx.pause()
        return

    model_path = ctx.pick_model_path(default_demo=True)
    if not model_path:
        ctx.pause()
        return

    default_audio = "input_data/demo_data/audio1.wav" if model_path == "models/demo_model/demo_model.ts" else ""
    audio_prompt = f"Reference audio path [{default_audio}]: " if default_audio else "Reference audio path: "
    audio_path = input(audio_prompt).strip() or default_audio
    if not audio_path or not os.path.exists(audio_path):
        print("[X] Invalid audio path.")
        ctx.pause()
        return

    output_name = input("Output name [generated]: ").strip() or "generated"
    duration = ctx.ask_int("Duration in seconds [30]: ", 30)
    ctx.generate_fn(model_path=model_path, audio_path=audio_path, output_name=output_name, duration=duration)
    ctx.pause()


def run_gui_stream(ctx):
    print("\n--- Multi-Model GUI Streaming ---")
    print("Launching GUI window. Close it when done.\n")
    try:
        ctx.launch_gui_fn()
    except Exception as e:
        print(f"[X] Failed to open GUI: {e}")
        traceback.print_exc()
    ctx.pause()


def generate_stream_menu(ctx):
    while True:
        print("\n" + "=" * 60)
        print("  Generate & Stream")
        print("=" * 60)
        print("  1) Generate audio from model")
        print("  2) Multi-model GUI streaming")
        print("  8) Back")
        print("  9) Home")
        print("  0) Exit")

        choice = ctx.ask_choice("\nChoose: ", {"1", "2", "8", "9", "0"})
        if choice == "1":
            run_generate_audio(ctx)
        elif choice == "2":
            run_gui_stream(ctx)
        elif choice == "8":
            return "BACK"
        elif choice == "9":
            return "HOME"
        elif choice == "0":
            return "EXIT"
