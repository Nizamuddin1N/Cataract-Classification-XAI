"""
main.py - Master script to run all phases of the Cataract Classification project.

Usage:
    python main.py           # Run all phases
    python main.py prepare   # Phase 1: Data preparation only
    python main.py train     # Phase 2: Training only
    python main.py evaluate  # Phase 3: Evaluation only
    python main.py gradcam   # Phase 4: Grad-CAM only
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))


def run_all():
    print("=" * 60)
    print("  CATARACT CLASSIFICATION USING EXPLAINABLE AI")
    print("  Running all phases...")
    print("=" * 60)

    from prepare_data import main as prepare
    from train import main as train
    from evaluate import main as evaluate
    from gradcam import main as gradcam

    prepare()
    train()
    evaluate()
    gradcam()

    print("\n" + "=" * 60)
    print("  🎉 ALL PHASES COMPLETE!")
    print("=" * 60)
    print("\nTo launch the demo app, run:")
    print("  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"

    if phase == "all":
        run_all()
    elif phase == "prepare":
        from prepare_data import main; main()
    elif phase == "train":
        from train import main; main()
    elif phase == "evaluate":
        from evaluate import main; main()
    elif phase == "gradcam":
        from gradcam import main; main()
    else:
        print(f"Unknown phase: {phase}")
        print("Usage: python main.py [all|prepare|train|evaluate|gradcam]")

