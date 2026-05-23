import os


def ensure_temp_exists():

    if not os.path.exists("temp"):
        os.makedirs("temp")