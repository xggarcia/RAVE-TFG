# -*- coding: utf-8 -*-
"""
Train a prior model using MSPrior for a pretrained RAVE model
"""
import os
import subprocess
import sys


def TrainPrior(rave_model_path, audio_path, prior_name="my_prior", config="decoder_only", output_path="models/user_model/prior"):
    """
    Train a prior model for a pretrained RAVE model using MSPrior.
    
    IMPORTANT: MSPrior requires RAVE checkpoint files (.ckpt), NOT exported .ts files.
    If you provide a .ts path, this function will try to find the original checkpoint.
    
    This function performs three steps:
    1. Preprocess: Encode audio dataset to latent representations using RAVE
    2. Train: Train the autoregressive prior on the latent dataset
    3. Export: Export the trained prior to a .ts file for use in Max/MSP or streaming
    
    Args:
        rave_model_path: Path to RAVE checkpoint (.ckpt) or run directory
                        If .ts is provided, will attempt to find checkpoint automatically
        audio_path: Path to the audio dataset folder (same data used to train RAVE)
        prior_name: Name for the prior model (default: "my_prior")
        config: MSPrior configuration (default: "decoder_only")
                Options: decoder_only, recurrent, encoder_decoder, encoder_decoder_continuous
        output_path: Base path for outputs (default: "models/user_model/prior")
    
    Returns:
        Path to exported prior .ts file if successful, None otherwise
    """
    print("=" * 60)
    print("  ENTRENAR PRIOR CON MSPRIOR")
    print("=" * 60)
    
    # Check if MSPrior is installed
    try:
        result = subprocess.run(
            ["msprior", "--help"],
            capture_output=True,
            text=True,
            shell=True
        )
        if result.returncode != 0:
            print("\n[X] Error: MSPrior no esta instalado")
            print("    Instalalo con: pip install acids-msprior")
            return None
    except FileNotFoundError:
        print("\n[X] Error: MSPrior no esta instalado")
        print("    Instalalo con: pip install acids-msprior")
        return None
    
    # Check if RAVE model exists
    if not os.path.exists(rave_model_path):
        print(f"\n[X] Error: Modelo RAVE no encontrado: {rave_model_path}")
        return None
    
    # MSPrior requires checkpoint files, not .ts exports
    # If user provided .ts file, try to find the checkpoint
    checkpoint_path = rave_model_path
    
    if rave_model_path.endswith('.ts'):
        print(f"\n[!] Advertencia: Proporcionaste un archivo .ts exportado")
        print(f"    MSPrior requiere archivos de checkpoint (.ckpt), no .ts")
        print(f"\n[*] Buscando checkpoint original...")
        
        # Try to find checkpoint from exported model path
        # Typical structure: models/user_model/exported_model/model.ts
        #                   models/user_model/checkpoints/last.ckpt
        model_dir = os.path.dirname(os.path.dirname(rave_model_path))
        checkpoint_dir = os.path.join(model_dir, "checkpoints")
        
        if os.path.exists(checkpoint_dir):
            # Look for checkpoint files
            ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
            if ckpt_files:
                # Prefer 'last.ckpt' or take the first one found
                if 'last.ckpt' in ckpt_files:
                    checkpoint_path = os.path.join(checkpoint_dir, 'last.ckpt')
                else:
                    checkpoint_path = os.path.join(checkpoint_dir, ckpt_files[0])
                print(f"[OK] Checkpoint encontrado: {checkpoint_path}")
            else:
                print(f"\n[X] Error: No se encontraron checkpoints en {checkpoint_dir}")
                print(f"    MSPrior solo funciona con modelos RAVE entrenados por ti,")
                print(f"    no con modelos descargados pre-entrenados.")
                print(f"\n    Necesitas:")
                print(f"    1. Entrenar tu propio modelo RAVE (Opcion B)")
                print(f"    2. Usar la carpeta de checkpoints, no el .ts exportado")
                return None
        else:
            print(f"\n[X] Error: No se encontro directorio de checkpoints")
            print(f"    Buscado en: {checkpoint_dir}")
            print(f"\n    MSPrior requiere el checkpoint original del entrenamiento RAVE")
            print(f"    Proporciona la ruta completa al checkpoint:")
            print(f"    --rave models/user_model/checkpoints/last.ckpt")
            return None
    
    # Check if audio path exists
    if not os.path.exists(audio_path):
        print(f"\n[X] Error: Carpeta de audio no encontrada: {audio_path}")
        return None
    
    # Create output directories
    preprocessed_path = os.path.join(output_path, "preprocessed_latents")
    training_path = os.path.join(output_path, "training")
    os.makedirs(preprocessed_path, exist_ok=True)
    os.makedirs(training_path, exist_ok=True)
    
    print(f"\n[1/3] PREPROCESAMIENTO: Codificando audio a espacio latente")
    print(f"      Checkpoint RAVE: {checkpoint_path}")
    print(f"      Dataset: {audio_path}")
    print(f"      Output: {preprocessed_path}")
    print("\n      Esto puede tardar varios minutos...")
    
    # Step 1: Preprocess - encode audio to latents
    preprocess_cmd = [
        "msprior", "preprocess",
        "--audio", audio_path,
        "--out_path", preprocessed_path,
        "--rave", checkpoint_path
    ]
    
    try:
        result = subprocess.run(preprocess_cmd, check=True, shell=True)
        print("\n[OK] Preprocesamiento completado")
    except subprocess.CalledProcessError as e:
        print(f"\n[X] Error durante preprocesamiento: {e}")
        return None
    
    print(f"\n[2/3] ENTRENAMIENTO: Entrenando prior autoregressivo")
    print(f"      Configuracion: {config}")
    print(f"      Nombre: {prior_name}")
    print(f"      Dataset latente: {preprocessed_path}")
    print("\n      Esto puede tardar desde minutos hasta horas dependiendo del dataset...")
    print("      Presiona Ctrl+C para detener el entrenamiento cuando estes satisfecho")
    
    # Step 2: Train the prior
    train_cmd = [
        "msprior", "train",
        "--config", config,
        "--db_path", preprocessed_path,
        "--name", prior_name,
        "--pretrained_embedding", checkpoint_path,
        "--out_path", training_path
    ]
    
    try:
        result = subprocess.run(train_cmd, shell=True)
        if result.returncode != 0 and result.returncode != -2:  # -2 is Ctrl+C
            print(f"\n[X] Error durante entrenamiento")
            return None
        print("\n[OK] Entrenamiento completado o detenido por usuario")
    except KeyboardInterrupt:
        print("\n[!] Entrenamiento interrumpido por usuario")
    except subprocess.CalledProcessError as e:
        print(f"\n[X] Error durante entrenamiento: {e}")
        return None
    
    # Find the trained model
    run_path = os.path.join(training_path, prior_name)
    if not os.path.exists(run_path):
        # Try to find any run directory
        if os.path.exists(training_path):
            runs = [d for d in os.listdir(training_path) if os.path.isdir(os.path.join(training_path, d))]
            if runs:
                run_path = os.path.join(training_path, runs[0])
                print(f"\n[!] Usando run encontrado: {run_path}")
            else:
                print(f"\n[X] No se encontro el modelo entrenado en {training_path}")
                return None
    
    print(f"\n[3/3] EXPORTACION: Exportando prior a TorchScript")
    print(f"      Run: {run_path}")
    
    # Step 3: Export the prior
    export_cmd = [
        "msprior", "export",
        "--run", run_path
    ]
    
    # Add --continuous flag if using continuous RAVE (not discrete)
    # We assume continuous by default (most common case)
    export_cmd.append("--continuous")
    
    try:
        result = subprocess.run(export_cmd, check=True, shell=True)
        print("\n[OK] Exportacion completada")
    except subprocess.CalledProcessError as e:
        print(f"\n[X] Error durante exportacion: {e}")
        print("\n[!] Si tu modelo RAVE usa configuracion 'discrete', intenta:")
        print(f"    msprior export --run {run_path}")
        print("    (sin el flag --continuous)")
        return None
    
    # Find exported .ts file
    exported_file = os.path.join(run_path, f"{prior_name}.ts")
    if os.path.exists(exported_file):
        print("\n" + "=" * 60)
        print("  ENTRENAMIENTO DE PRIOR COMPLETADO")
        print("=" * 60)
        print(f"\n[OK] Prior exportado: {exported_file}")
        print("\n[*] Como usar el prior:")
        print("    1. En Max/MSP con nn~:")
        print(f"       [nn~ {exported_file}]")
        print("\n    2. Junto con RAVE en Max/MSP:")
        print(f"       [nn~ {rave_model_path}] -> encode")
        print(f"       [nn~ {exported_file}] -> generate latents")
        print(f"       [nn~ {rave_model_path}] -> decode")
        print("\n" + "=" * 60)
        return exported_file
    else:
        print(f"\n[!] Advertencia: Archivo exportado no encontrado en la ubicacion esperada")
        print(f"    Busca archivos .ts en: {run_path}")
        return run_path


def main():
    """Test function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a prior for a RAVE model using MSPrior")
    parser.add_argument("--rave", required=True, help="Path to RAVE checkpoint (.ckpt) or run directory")
    parser.add_argument("--audio", required=True, help="Path to audio dataset folder")
    parser.add_argument("--name", default="my_prior", help="Name for the prior model")
    parser.add_argument("--config", default="decoder_only", 
                       choices=["decoder_only", "recurrent", "encoder_decoder", "encoder_decoder_continuous"],
                       help="MSPrior configuration")
    parser.add_argument("--output", default="models/user_model/prior", help="Output path for prior")
    
    args = parser.parse_args()
    
    TrainPrior(
        rave_model_path=args.rave,
        audio_path=args.audio,
        prior_name=args.name,
        config=args.config,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
