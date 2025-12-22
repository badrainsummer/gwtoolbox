import os
import time
import requests
from zipfile import ZipFile
from tqdm import tqdm
import shutil

def download_and_extract_zip(url, extract_to='.'):
    """
    从 Zenodo 下载 zip 文件并解压到指定目录（默认当前目录）
    自动处理 429 限流，带下载进度条。
    下载后如果 zip 内生成了一个大文件夹，会把子文件/子文件夹移动到 extract_to 并删除原文件夹。
    """
    os.makedirs(extract_to, exist_ok=True)
    filename = url.split('/')[-1].split('?')[0]  # 去掉 URL 参数
    zip_path = os.path.join(extract_to, filename)

    # 下载文件
    if not os.path.exists(zip_path):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                with requests.get(url, headers=headers, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    total = int(r.headers.get('content-length', 0))
                    with open(zip_path, 'wb') as f, tqdm(
                        total=total, unit='B', unit_scale=True, desc=filename
                    ) as pbar:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                print(f"Downloaded {filename} successfully.")
                break
            except requests.exceptions.HTTPError as e:
                if r.status_code == 429:
                    wait = (attempt + 1) * 5
                    print(f"Too many requests. Waiting {wait} seconds before retrying...")
                    time.sleep(wait)
                else:
                    raise e
            except requests.exceptions.RequestException as e:
                print(f"Download failed: {e}. Retrying...")
                time.sleep(5)
        else:
            raise RuntimeError("Failed to download file after multiple retries.")

    # 解压
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {filename} to {extract_to}")

    # 如果解压后生成了一个大文件夹，把子内容搬出来
    extracted_folder_name = filename.replace('.zip','')
    extracted_folder_path = os.path.join(extract_to, extracted_folder_name)
    if os.path.isdir(extracted_folder_path):
        for item in os.listdir(extracted_folder_path):
            src = os.path.join(extracted_folder_path, item)
            dst = os.path.join(extract_to, item)
            # 如果目标已存在，改名或覆盖（这里选择覆盖）
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
        shutil.rmtree(extracted_folder_path)
        print(f"Moved contents from {extracted_folder_name} to {extract_to} and removed the folder.")

    # 可选择删除 zip 文件
    os.remove(zip_path)

    # mac删除__MACOSX文件夹
    macosx_path = os.path.join(extract_to, '__MACOSX')
    if os.path.exists(macosx_path):
        shutil.rmtree(macosx_path)
        print("Removed __MACOSX folder")

if __name__ == "__main__":
    ZENODO_URL = "https://zenodo.org/record/17962656/files/Supplementary Materials.zip"
    download_and_extract_zip(ZENODO_URL, extract_to='.')
