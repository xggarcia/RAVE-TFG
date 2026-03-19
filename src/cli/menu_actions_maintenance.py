def run_clean_user_data(ctx):
    print("\n--- Clean User Data ---")
    if ctx.ask_yes_no("This will remove generated user data. Continue", default_yes=False):
        ctx.clean_fn()
    ctx.pause()


def show_help_about(ctx):
    print("\n--- Help / About ---")
    print("RAVE-TFG interactive menu:")
    print("- Generate & Stream: run inference and real-time tools")
    print("- Data & Training: preprocess, train, export, full workflow, train prior")
    print("- Maintenance: cleanup and utility information")
    print("\nTip: Use Quick mode for safe defaults, Advanced mode for full control.")
    ctx.pause()


def maintenance_menu(ctx):
    while True:
        print("\n" + "=" * 60)
        print("  Maintenance")
        print("=" * 60)
        print("  1) Clean user data")
        print("  2) Help / About")
        print("  8) Back")
        print("  9) Home")

        choice = ctx.ask_choice("\nChoose: ", {"1", "2", "8", "9"})
        if choice == "1":
            run_clean_user_data(ctx)
        elif choice == "2":
            show_help_about(ctx)
        elif choice == "8":
            return "BACK"
        elif choice == "9":
            return "HOME"
