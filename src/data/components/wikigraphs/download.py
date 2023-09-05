import os
import sys
import shutil
import subprocess

from src import utils

log = log = utils.get_pylogger(__name__)

def wikigraphs_exists(data_dir) -> bool:
    wikitext_103_raw = os.path.exists(os.path.join(data_dir, 'wikitext-103-raw', 'wiki.train.raw'))
    wikitext_103 = os.path.exists(os.path.join(data_dir, 'wikitext-103', 'wiki.train.tokens'))
    freebase = os.path.exists(os.path.join(data_dir, 'freebase', 'max256', 'train.gz'))
    return True if wikitext_103_raw and wikitext_103 and freebase else False
    

def download_wikigraphs(data_dir):
    if not wikigraphs_exists(data_dir):
        try:
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
                
            BASE_DIR = data_dir

            # wikitext-103
            TARGET_DIR = os.path.join(BASE_DIR, 'wikitext-103')
            os.makedirs(TARGET_DIR, exist_ok=True)
            subprocess.run(['wget', 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip', '-P', TARGET_DIR])
            subprocess.run(['unzip', os.path.join(TARGET_DIR, 'wikitext-103-v1.zip'), '-d', TARGET_DIR])
            # Move files programmatically
            wikitext_103_raw_dir = os.path.join(TARGET_DIR, 'wikitext-103')
            for file in os.listdir(wikitext_103_raw_dir):
                src = os.path.join(wikitext_103_raw_dir, file)
                dst = os.path.join(TARGET_DIR, file)
                shutil.move(src, dst)
            # Remove the extracted unused folder and zip
            subprocess.run(['rm', '-rf', os.path.join(TARGET_DIR, 'wikitext-103'), os.path.join(TARGET_DIR, 'wikitext-103-v1.zip')])

            # wikitext-103-raw
            TARGET_DIR = os.path.join(BASE_DIR, 'wikitext-103-raw')
            os.makedirs(TARGET_DIR, exist_ok=True)
            subprocess.run(['wget', 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip', '-P', TARGET_DIR])
            subprocess.run(['unzip', os.path.join(TARGET_DIR, 'wikitext-103-raw-v1.zip'), '-d', TARGET_DIR])
            # Move files programmatically
            wikitext_103_raw_dir = os.path.join(TARGET_DIR, 'wikitext-103-raw')
            for file in os.listdir(wikitext_103_raw_dir):
                src = os.path.join(wikitext_103_raw_dir, file)
                dst = os.path.join(TARGET_DIR, file)
                shutil.move(src, dst)
            # Remove the extracted unused folder and zip
            subprocess.run(['rm', '-rf', os.path.join(TARGET_DIR, 'wikitext-103-raw'), os.path.join(TARGET_DIR, 'wikitext-103-raw-v1.zip')])

            # processed freebase graphs
            FREEBASE_TARGET_DIR = data_dir
            os.makedirs(os.path.join(FREEBASE_TARGET_DIR, 'packaged'), exist_ok=True)
            subprocess.run(['wget', '--no-check-certificate', 'https://docs.google.com/uc?export=download&id=1uuSS2o72dUCJrcLff6NBiLJuTgSU-uRo', '-O', os.path.join(FREEBASE_TARGET_DIR, 'packaged', 'max256.tar')])
            subprocess.run(['wget', '--no-check-certificate', 'https://docs.google.com/uc?export=download&id=1nOfUq3RUoPEWNZa2QHXl2q-1gA5F6kYh', '-O', os.path.join(FREEBASE_TARGET_DIR, 'packaged', 'max512.tar')])
            subprocess.run(['wget', '--no-check-certificate', 'https://docs.google.com/uc?export=download&id=1uuJwkocJXG1UcQ-RCH3JU96VsDvi7UD2', '-O', os.path.join(FREEBASE_TARGET_DIR, 'packaged', 'max1024.tar')])

            for version in ['max1024', 'max512', 'max256']:
                output_dir = os.path.join(FREEBASE_TARGET_DIR, 'freebase', version)
                os.makedirs(output_dir, exist_ok=True)
                subprocess.run(['tar', '-xvf', os.path.join(FREEBASE_TARGET_DIR, 'packaged', f'{version}.tar'), '-C', output_dir])

            # Remove the packaged files
            subprocess.run(['rm', '-rf', os.path.join(FREEBASE_TARGET_DIR, 'packaged')])
        except:
            log.error("""You probably either don't have the necessary write permission \
                or don't have packages like 'unzip' installed. Try the 'download_wikigraphs.sh'\
                script and make sure to run in sudo mode and install all required packages.""")
            sys.exit()
    else:
        log.info("Raw dataset exists - skipping download.")
