

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', '0', 'no', 'off'}:
        return False
    if value.lower() in {'true', '1', 'yes', 'on'}:
        return True
    raise ValueError(f'Invalid boolean value: {value}')

def make_method_label(method, error_weight, random_assign, scale):
    label = f"{method}_error{error_weight}"
    if random_assign:
        label += "_rnd"
    if scale:
        label += "_scl"
    return label