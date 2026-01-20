import json
from pathlib import Path
from cademas.validation.schemas import ADMMetadata


def load_adm_metadata(path: str):
    data = json.loads(Path(path).read_text())
    return [ADMMetadata(**adm) for adm in data]