import subprocess, zipfile, pathlib, urllib.request, shutil

def download_trashnet(dest="data/trashnet"):
    url = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
    dest = pathlib.Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    zip_path = dest.with_suffix(".zip")
    print("▶ downloading TrashNet ...")
    urllib.request.urlretrieve(url, zip_path)
    print("▶ extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest.parent)
    extracted = dest.parent / "dataset-resized"
    if dest.exists():
        shutil.rmtree(dest)
    extracted.rename(dest)
    zip_path.unlink()
    print("✔ TrashNet ready:", dest)

if __name__ == "__main__":
    download_trashnet()