def get_differences(obj1, obj2, path=""):  # noqa: D100
    """
    Compare two dictionaries or None and return differences as a single string
    of the format: key: old value -> new value, separated by " | ".
    """  # noqa: D205, D212
    if obj1 is None and obj2 is None:
        return ""

    if obj1 is None:
        return f"{path}: None -> {obj2}" if path else f"root: None -> {obj2}"

    if obj2 is None:
        return f"{path}: {obj1} -> None" if path else f"root: {obj1} -> None"

    differences = []

    # Handle dictionaries
    all_keys = set(obj1.keys()).union(obj2.keys())
    for key in all_keys:
        current_path = f"{path}.{key}" if path else str(key)
        if key in obj1 and key in obj2:
            if obj1[key] != obj2[key]:
                if isinstance(obj1[key], dict) and isinstance(obj2[key], dict):
                    nested_diff = get_differences(obj1[key], obj2[key], current_path)
                    if nested_diff:
                        differences.append(nested_diff)
                else:
                    differences.append(f"{current_path}: {obj1[key]} -> {obj2[key]}")
        elif key in obj1:
            differences.append(f"{current_path}: {obj1[key]} -> None")
        else:
            differences.append(f"{current_path}: None -> {obj2[key]}")

    return "\n    ".join(differences)
