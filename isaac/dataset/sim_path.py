import os

FRAMEWORK_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
    #os.path.dirname(__file__): the path of the Python file this code is in
)


def sim_framework_path(*args) -> str:
    """
    Abstraction from os.path.join()
    Builds absolute paths from relative path strings with SIM_FRAMEWORK/ as root.
    If args already contains an absolute path, it is used as root for the subsequent joins
    Args:
        *args:

    Returns:
        absolute path

    """
    return os.path.abspath(os.path.join(FRAMEWORK_DIR, *args))
