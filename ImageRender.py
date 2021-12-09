from Obfuscator import obfuscator
from DataManager import CheckFileExist
from wand.color import Color
from wand.image import Image
from wand.drawing import Drawing
import pandas as pd
import csv
import os

# data -> Images + CSV
# If Imagename=="" it will be the textstring in the image
def csv_obf_img(imagename="", filename="data/dataset.csv",
                data="data/HASOC_english_dataset/HASOC2019+2020+CONAN-ASCII.csv", imagepath="data/images/",
                obfuscationmethod=2):
    if isinstance(data, list):
        hatespeechdata = data
    else:
        hatespeechdata = pd.read_csv(data, encoding="utf_8")
        hatespeechdata = hatespeechdata.loc[:, "Tweet"]
    CheckFileExist(filename)

    with open(filename, "x", encoding="utf_16") as file:
        file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_writer.writerow(["Tweet", "obfuscated_Tweet"])
        j = 0
        img_count = len(hatespeechdata)
        for i in hatespeechdata:
            obfuscated_Satz = obfuscator(str(i),obfuscationmethod) #2 = Einzelzeichen
            #obfuscated_Satz = str(i.lower())######################## no obfuscation & kleinbuchstaben
            if imagename =="":
                imagename = obfuscated_Satz
                j=""
            createImage(obfuscated_Satz, j, name=imagename, path=imagepath)
            print("image",j,"/",img_count)
            file_writer.writerow([i, obfuscated_Satz])
            j += 1
    return j

def createImage(text, namesuffix, name, path):
    filename = os.path.join(path, name + str(namesuffix) + ".png")
    #font = "C:/Users/Lenard/PycharmProjects/Bachelor-Code/unifont-13.0.05.ttf" ################<<<<<<< FONT total PATH
    font = "C:/Users/lenar/Documents/deobfuscator-main/deobfuscator-main/unifont-13.0.05.ttf"
    font = "/home/ba/.fonts/unifont-13.0.05.ttf"
    font_size = 31.0 #32.0

    # Temporary adding very tall String, to make sure the text in the image will be centered
    text += " S̨̥̫͎̭ͯ̿̔̀ͅņ"
    with Drawing() as draw:
        with Image(width=1, height=1) as img_0: #Temporary create Image to calculate length of  S̨̥̫͎̭ͯ̿̔̀ͅņ
            draw.font = font
            draw.font_size = font_size
            width = int(draw.get_font_metrics(img_0,text)[4]) #Länge des Strings berechnen
            chop = int(draw.get_font_metrics(img_0,"S̨̥̫͎̭ͯ̿̔̀ͅņ")[4])
            height = int(draw.get_font_metrics(img_0,text)[5]) #33.0
            print("/n/n----------height",height)
        with Image(width=width, height=height+height//2, background=Color('white')) as img:  #height=height+height//2
            draw.font = font
            draw.font_size = font_size
            draw.text(0, height, text)   #variable in Höhe..   #ก้้้้้้้้้้้้้้้้้้้้ #ƒ #S̨̥̫͎̭ͯ̿̔̀ͅ deshalb wird hier einheitliche Höhe festgelegt
            draw(img)
            img.trim()
            img.chop(chop, img.height, img.width-chop, img.height) # chop the S̨̥̫͎̭ͯ̿̔̀ͅņ
            #img.chop(0, 32, 0, -31)  # chop last row of pixels #for 33-1 -> 32

            #img.chop(width=img.width, height=1, x=0, y=1) #funz iwie nicht, aber in Datamanager gelöst
            img.save(filename=filename)

#data.csv -> Image
def newImagesfromCsv(imagename = "imageTNR",font = "Times New Roman",font_size=20.0,data="text-data-1"):
    datafile = "data/" + data + ".csv"
    present_df = pd.read_csv(datafile,encoding="utf_16")
    obfTweet_column = present_df["obfuscated_Tweet"]
    i=0
    for obf_Tweet in obfTweet_column:
        createImage(obf_Tweet,i,imagename,font)
        i+=1


