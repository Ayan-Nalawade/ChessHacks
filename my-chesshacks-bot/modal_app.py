import modal

app = modal.App("chesshacks-bot")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "annotated-doc==0.0.4",
        "annotated-types==0.7.0",
        "anyio==4.11.0",
        "chess==1.11.2",
        "click==8.1.8",
        "exceptiongroup==1.3.0",
        "fastapi==0.121.2",
        "h11==0.16.0",
        "idna==3.11",
        "pydantic==2.12.4",
        "pydantic_core==2.41.5",
        "python-chess==1.999",
        "sniffio==1.3.1",
        "starlette==0.49.3",
        "typing-inspection==0.4.2",
        "typing_extensions==4.15.0",
        "uvicorn==0.38.0",
        "numpy==2.3.0",  
        "torch==2.9.1",
        "torchvision==0.24.1",
        "huggingface-hub==0.26.2",
        "tqdm==4.67.1",
    )
    .add_local_dir("src", remote_path="/root/src")
    .add_local_file("download_model.py", remote_path="/root/download_model.py")
)


# Create a persistent volume to store trained models
volume = modal.Volume.from_name("chess-models", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/model_storage": volume},
)
def build_dataset(
    pattern: str = "24-12-*/*/*.pgn.gz",
    position_cap: int = 50000,
    max_games: int | None = None,
    max_positions: int | None = None,
    output_path: str = "/model_storage/datasets/fishtest.jsonl",
):
    import sys
    from pathlib import Path

    sys.path.append("/root")
    from download_model import generate_samples

    cache_dir = Path("/model_storage/raw")
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    total = generate_samples(
        pattern=pattern,
        cache_dir=cache_dir,
        output=output,
        max_games=max_games,
        max_positions=max_positions,
        position_cap=position_cap,
        overwrite=True,
    )
    return {"samples_created": total, "output": str(output)}


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    volumes={"/model_storage": volume},
)
def train_model(
    dataset_path: str = "/model_storage/datasets/fishtest.jsonl",
    epochs: int = 3,
    batch_size: int = 256,
    steps_per_epoch: int = 500,
    lr: float = 3e-4,
    output_name: str = "value_net.pt",
):
    import sys
    from pathlib import Path

    sys.path.append("/root/src")
    from src.training import train_supervised

    dataset = Path(dataset_path)
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset}")

    output_path = Path("/model_storage/checkpoints") / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_path = train_supervised(
        data_paths=[dataset],
        output_path=output_path,
        epochs=epochs,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        lr=lr,
    )

    volume.commit()
    return {
        "status": "success",
        "model_path": str(result_path),
    }


@app.function(
    image=image,
    volumes={"/model_storage": volume},
)
def download_trained_model():
    """
    Download the trained model from Modal's persistent storage
    
    Returns:
        The model file as bytes
    """
    import os
    
    model_path = "/model_storage/checkpoints/best.pth.tar"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found at {model_path}")
    
    with open(model_path, "rb") as f:
        model_data = f.read()
    
    return model_data


@app.function(
    image=image,
    volumes={"/model_storage": volume},
)
def list_saved_models():
    """
    List all saved model checkpoints in Modal storage
    
    Returns:
        List of model filenames
    """
    import os
    
    checkpoint_dir = "/model_storage/checkpoints/"
    
    if not os.path.exists(checkpoint_dir):
        return {"models": [], "message": "No checkpoint directory found"}
    
    models = os.listdir(checkpoint_dir)
    return {
        "models": models,
        "checkpoint_dir": checkpoint_dir
    }


@app.local_entrypoint()
def main(action: str = "train"):
    """
    Main entrypoint for Modal operations
    
    Usage:
        modal run modal_app.py --action train           # Train the model
        modal run modal_app.py --action list            # List saved models
        modal run modal_app.py --action download        # Download trained model
    """
    import sys
    
    if action == "train":
        print("Starting model training on Modal GPU...")
        result = train_model.remote(
            dataset_path="/model_storage/datasets/fishtest.jsonl",
            epochs=3,
            batch_size=256,
            steps_per_epoch=500,
            lr=3e-4,
        )
        print(f"\nTraining complete!")
        print(f"Status: {result['status']}")
        print(f"Model saved to: {result['model_path']}")
        
    elif action == "dataset":
        print("Building dataset slice on Modal...")
        result = build_dataset.remote(
            pattern="24-12-*/*/*.pgn.gz",
            position_cap=50000,
        )
        print(f"Dataset stored at: {result['output']} ({result['samples_created']} samples)")

    elif action == "list":
        print("Listing saved models...")
        result = list_saved_models.remote()
        print(f"\nSaved models: {result['models']}")
        
    elif action == "download":
        print("Downloading trained model...")
        model_data = download_trained_model.remote()
        
        # Save to local file
        output_path = "trained_model.pth.tar"
        with open(output_path, "wb") as f:
            f.write(model_data)
        print(f"Model downloaded to: {output_path}")
        print(f"Size: {len(model_data) / (1024*1024):.2f} MB")
        
    else:
        print(f"Unknown action: {action}")
        print("Valid actions: dataset, train, list, download")
        sys.exit(1)
