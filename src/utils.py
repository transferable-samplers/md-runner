from pathlib import Path
from omegaconf import DictConfig

def get_md_output_dir(cfg: DictConfig, seq_name: str) -> Path:  
    return Path(f"{cfg.paths.data_dir}/md/{seq_name}_{cfg.temperature}_{cfg.frame_interval}_{cfg.frames_per_chunk}") 