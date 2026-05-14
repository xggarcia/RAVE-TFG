from src.core.clean import CleanUserData
from src.core.export import ExportModel
from src.core.generate import UseModel
from src.core.preprocess import PreprocessDataset
from src.core.train import TrainModel
from src.core.workflow import train_workflow

from .menu_actions_generate import generate_stream_menu
from .menu_actions_maintenance import maintenance_menu
from .menu_actions_training import data_training_menu
from .menu_helpers import MenuContext


def _build_menu_context():
    return MenuContext(
        preprocess_fn=PreprocessDataset,
        train_fn=TrainModel,
        export_fn=ExportModel,
        generate_fn=UseModel,
        workflow_fn=train_workflow,
        clean_fn=CleanUserData,
    )


def interactive_menu():
    """Interactive menu coordinator that delegates category behavior by module."""
    ctx = _build_menu_context()

    while True:
        print("\n" + "=" * 60)
        print("  RAVE-TFG - Main Menu")
        print("=" * 60)
        print("  1) Generate & Stream")
        print("  2) Data & Training")
        print("  3) Maintenance")
        print("  0) Exit")

        choice = ctx.ask_choice("\nChoose: ", {"1", "2", "3", "0"})
        if choice == "1":
            result = generate_stream_menu(ctx)
            if result == "EXIT":
                print("\nGoodbye!")
                break
        elif choice == "2":
            result = data_training_menu(ctx)
            if result == "EXIT":
                print("\nGoodbye!")
                break
        elif choice == "3":
            result = maintenance_menu(ctx)
            if result == "EXIT":
                print("\nGoodbye!")
                break
        elif choice == "0":
            print("\nGoodbye!")
            break
