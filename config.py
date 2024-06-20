from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 25,
        "lr": 10**-4, 
        "seq_len": 10, 
        "d_model": 512, 
        "num_features": 7,
        "model_folder": "weights", 
        "model_basename": "s_model",
        "preload": None,
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)
