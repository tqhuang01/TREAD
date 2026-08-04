import warnings

warnings.filterwarnings(
    "ignore",
    message="Only reconstruction head is pre-trained. Classification and forecasting heads must be fine-tuned."
)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from momentfm import MOMENTPipeline
import json


def load_moment_model(
        model_dir: str,
        context_length: int = 512,
        forecast_horizon: int = None
) -> MOMENTPipeline:
    config_path = model_dir + "/config.json"
    config = json.load(open(config_path, encoding="utf-8"))
    config["seq_len"] = context_length
    json.dump(config, open(config_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    model = MOMENTPipeline.from_pretrained(
        model_dir,
        model_kwargs={
            "task_name": "forecasting",
            "forecast_horizon": forecast_horizon,
            "head_dropout": 0.1,
            "weight_decay": 0,
            "freeze_encoder": True,
            "freeze_embedder": True,
            "freeze_head": False,
        },
        local_files_only=True,
    )
    model.init()
    return model
