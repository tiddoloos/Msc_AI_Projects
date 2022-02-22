
from PIL import Image
import glob

# save image to file with correct label
def sort_images(path):
    for i in range(0,10,1):
        files = glob.glob(path + str(i) +"/*.png")
        for filename in files: #assuming gif
            print(filename)
            im=Image.open(filename)
            if im.size == (32,32):
                print(im.size)
                im.save('32x32/'+filename)
            elif im.size == (48,48):
                im.save('48x48/'+filename)
            elif im.size == (64,64):
                im.save('64x64/'+filename)
            im.close()

trainPath = ("mnist-varres/train/")
sort_images(trainPath)
testPath = ("mnist-varres/test/")
sort_images(testPath)

# import shutil
# import os
 
# file_source = 'Path/Of/Directory'
# file_destination = 'Path/Of/Directory'
 
# get_files = os.listdir(file_source)
 
# for g in get_files:
#     shutil.move(file_source + g, file_destination