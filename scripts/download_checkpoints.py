import os
import urllib.request
import sys

def download_file(url, filename):
    print(f"Downloading {url} to {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print("Done.")
    except Exception as e:
        print(f"Failed to download: {e}")

def main():
    os.makedirs("checkpoints", exist_ok=True)
    
    # Heaviest Network (Pretrained initialization)
    # url_heaviest = "https://miil-public-eu.oss-eu-central-1.aliyuncs.com/public/HardCoReNAS/w_heaviest_d89ee05d.pth"
    # download_file(url_heaviest, "checkpoints/w_heaviest.pth")
    
    # Supernet (One-shot model - Ready for search)
    url_supernet = "https://github.com/thanquannguyen/HardCoReNAS_pi/raw/main/checkpoints/w_supernet.pth"
    download_file(url_supernet, "checkpoints/w_supernet.pth")
    
    # HardCoRe-NAS A (Reference model for compression demo)
    url_nas_a = "https://github.com/thanquannguyen/HardCoReNAS_pi/raw/main/checkpoints/hardcorenas_a.pth"
    download_file(url_nas_a, "checkpoints/hardcorenas_a.pth")

if __name__ == "__main__":
    main()
