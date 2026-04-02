import torch


def resolve_device(requested: str, *, strict: bool = False) -> str:
    requested_norm = (requested or "cpu").strip().lower()

    if requested_norm.startswith("cuda"):
        if torch.cuda.is_available():
            return requested
        message = (
            f"Requested device '{requested}' but CUDA is not available. "
            "Falling back to CPU."
        )
        if strict:
            raise RuntimeError(message)
        print(f"[device warning] {message}")
        return "cpu"

    if requested_norm == "cpu":
        return "cpu"

    # Keep unknown device values as-is to preserve backwards compatibility.
    return requested
