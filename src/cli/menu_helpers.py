import os


class MenuContext:
    """Shared helpers and operation callables for interactive menu actions."""

    def __init__(
        self,
        preprocess_fn,
        train_fn,
        export_fn,
        generate_fn,
        workflow_fn,
        clean_fn,
    ):
        self.preprocess_fn = preprocess_fn
        self.train_fn = train_fn
        self.export_fn = export_fn
        self.generate_fn = generate_fn
        self.workflow_fn = workflow_fn
        self.clean_fn = clean_fn

    @staticmethod
    def ask_path(prompt):
        """Read a path from the user and normalize separators for the current OS."""
        return os.path.normpath(input(prompt).strip())

    def ask_choice(self, prompt, valid_choices):
        while True:
            choice = input(prompt).strip()
            if choice in valid_choices:
                return choice
            print(f"[X] Invalid choice. Valid options: {', '.join(valid_choices)}")

    def ask_int(self, prompt, default):
        while True:
            raw = input(prompt).strip()
            if not raw:
                return default
            try:
                return int(raw)
            except ValueError:
                print("[X] Please enter an integer value.")

    def ask_yes_no(self, prompt, default_yes=True):
        default_label = "Y/n" if default_yes else "y/N"
        while True:
            raw = input(f"{prompt} [{default_label}]: ").strip().lower()
            if not raw:
                return default_yes
            if raw in {"y", "yes"}:
                return True
            if raw in {"n", "no"}:
                return False
            print("[X] Please answer with y or n.")

    @staticmethod
    def pause():
        input("\nPress Enter to continue...")

    def model_mode_prompt(self):
        print("\nSelect setup mode:")
        print("  1) Quick (recommended defaults)")
        print("  2) Advanced (custom parameters)")
        print("  9) Cancel")
        return self.ask_choice("Choose: ", {"1", "2", "9"})

    @staticmethod
    def find_exported_model():
        export_dir = "models/user_model/exported_model"
        if not os.path.exists(export_dir):
            return None
        ts_files = [f for f in os.listdir(export_dir) if f.endswith(".ts")]
        if not ts_files:
            return None
        return os.path.join(export_dir, ts_files[0])

    @staticmethod
    def find_rave_checkpoint():
        checkpoint_dir = "models/user_model/checkpoints"
        if not os.path.exists(checkpoint_dir):
            return None
        ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
        if not ckpt_files:
            return None
        if "last.ckpt" in ckpt_files:
            return os.path.join(checkpoint_dir, "last.ckpt")
        return os.path.join(checkpoint_dir, ckpt_files[0])

    def pick_model_path(self, default_demo=True):
        print("\nModel source:")
        print("  1) Demo model")
        print("  2) User exported model (.ts)")
        default = "1" if default_demo else "2"
        choice = input(f"Choose [default {default}]: ").strip() or default
        if choice == "1":
            return "models/demo_model/demo_model.ts"

        custom = input("Model .ts path (leave empty for auto-detect): ").strip()
        if custom:
            if os.path.exists(custom):
                return custom
            print(f"[X] Model not found: {custom}")
            return None

        model_path = self.find_exported_model()
        if model_path:
            print(f"[OK] Using detected exported model: {model_path}")
            return model_path

        print("[X] No exported .ts model found in models/user_model/exported_model")
        return None
