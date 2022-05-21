from urllib.request import urlretrieve
import os

def download(url, file_path):
    dirname = os.path.dirname(file_path)
    if os.path.exists(file_path):
        print("The file already exists. Skip downloading")
        return
    if not os.path.exists(dirname):
        os.system("mkdir -p "+dirname)
    try:
        print(f"Downloading {url} to {dirname}")
        urlretrieve(url, file_path)
    except Exception:
        raise RuntimeError("downloading data error")
        