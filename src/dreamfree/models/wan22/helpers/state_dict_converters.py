def wan_video_vae_state_dict_converter(state_dict):
    converted = {}
    if "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    for name, value in state_dict.items():
        converted[f"model.{name}"] = value
    return converted


def wan_video_dit_state_dict_converter(state_dict):
    converted = {}
    for name, value in state_dict.items():
        if name.startswith("vace"):
            continue
        if name.split(".")[0] in [
            "pose_patch_embedding",
            "face_adapter",
            "face_encoder",
            "motion_encoder",
        ]:
            continue
        key = name[6:] if name.startswith("model.") else name
        converted[key] = value
    return converted
