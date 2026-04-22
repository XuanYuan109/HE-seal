import os
import zipfile

print("📦 Downloading dataset from Kaggle...")

os.makedirs("data", exist_ok=True)

# 下载
os.system("kaggle datasets download -d xuanyuan/images-for-he-seal -p data/")

# 解压
for file in os.listdir("data"):
    if file.endswith(".zip"):
        path = os.path.join("data", file)
        print(f"📂 Extracting {file}...")
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall("data")

print("✅ Done! Data ready in ./data/")