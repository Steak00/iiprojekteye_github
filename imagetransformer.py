from PIL import Image
from pillow_heif import register_heif_opener
import sys
import os
from pathlib import Path

register_heif_opener()

listdir = os.listdir(sys.argv[1])
cwd = os.getcwd()

Path(cwd + '\\' + sys.argv[1] + '\\converted').mkdir(parents=True, exist_ok=True)

for i in listdir[:5]:
    try:
        print(cwd + '\\' + sys.argv[1] + '\\' + i)
        image = Image.open(cwd + '\\' + sys.argv[1] + '\\' + i)
        newfilename = i.split('.')[0] + '.jpg'
        image = image.save(sys.argv[1] + '\converted\\' + newfilename)
        print('converted ' + newfilename)
    except FileNotFoundError:
        print("FileNotFoundError")
        continue
    except:
        print("Anderer Fehler")
        continue

