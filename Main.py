from ImageRender import csv_obf_img
from DataManager import clearDirectory, SplitData
from Evaluation import applyCompareModels
from ImageRender import createImage
#import re

data = "data/HASOC_english_dataset/HASOC2019+2020+CONAN-Wordset.csv" # - Kopie.csv"
csvFile = "BrandNewModel/dataset.csv"
imagepath = "data/images/" # use / in the end pls
imagename = "image"
#evaluationFile = "BrandNewModel\evaluationIMGspace.csv" == csvFile ?

createImage(" S̨̥̫͎̭ͯ̿̔̀ͅņ","-TEST","Bild",imagepath)


# create Images & Output csv, clear BrandNewModel-Folder
if __name__ == "__main__":
    clearDirectory(path="BrandNewModel/*")
    data, evaldata = SplitData(data, trainevalPath="BrandNewModel/dataset")
    imagecount = csv_obf_img(imagename=imagename,
                        filename=csvFile, # wird erstellt
                        data=data, # raw data
                        #data="data\IWG_hatespeech_public-master\BITTRASH_trash-data.csv",
                        imagepath=imagepath,
                        obfuscationmethod=2 #2 = Einzelzeichen
                            )

# Train Model (takes tons of time and/or computational powerr)
    #runfile('C:/Users/lenar/Documents/deobfuscator-main/deobfuscator-main/FinalModelCRNN.py', wdir='C:/Users/lenar/Documents/deobfuscator-main/deobfuscator-main')

# Apply Models (trained one + baselines)
#applyCompareModels(imagepath,imagename,csvFile)

# Evaluate Models further
