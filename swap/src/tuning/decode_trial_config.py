def decode_special_value(key, value):
    # If the value is already a structured object (list/tuple/dict),
    # do not try to use it as a mapping key.
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return value

    if key == "model.attention_resolutions":
        mapping = {
            "8": [8],
            "16": [16],
            "32": [32],
            "64": [64],
            "8-16": [8, 16],
            "16-32": [16, 32],
            "32-64": [32, 64],
            "16-32-64": [16, 32, 64],
        }
        if value not in mapping:
            raise ValueError(f"Unknown attention_resolutions token: {value}")
        return mapping[value]

    if key == "model.channel_mult":
        mapping = {
            # 128 / 256 style
            "1-2-4": [1, 2, 4],
            "1-2-4-8": [1, 2, 4, 8],
            "1-2-2-4": [1, 2, 2, 4],
            "1-1-2-3-4": [1, 1, 2, 3, 4],
            "1-2-4-4": [1, 2, 4, 4],

            # 512 style
            "1-1-2-2-4-4": [1, 1, 2, 2, 4, 4],
            "1-2-2-4-4-8": [1, 2, 2, 4, 4, 8],
            "1-1-2-4-4-8": [1, 1, 2, 4, 4, 8],
            "1-2-4-4-8-8": [1, 2, 4, 4, 8, 8],
        }
        if value not in mapping:
            raise ValueError(f"Unknown channel_mult token: {value}")
        return mapping[value]

    if key == "pretrain.betas":
        mapping = {
            "0.9-0.99": [0.9, 0.99],
            "0.9-0.999": [0.9, 0.999],
        }
        if value not in mapping:
            raise ValueError(f"Unknown pretrain.betas token: {value}")
        return mapping[value]

    return value