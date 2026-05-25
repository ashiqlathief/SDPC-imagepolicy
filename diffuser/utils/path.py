import os

def project_path(*parts):
    """
    Returns an absolute path relative to the project root.
    """
    current = os.path.abspath(os.path.dirname(__file__))

    while current != "/":
        if "isaac" in os.listdir(current):
            return os.path.join(current, *parts)
        current = os.path.dirname(current)

    raise RuntimeError("Could not find project root containing 'isaac' folder.")
