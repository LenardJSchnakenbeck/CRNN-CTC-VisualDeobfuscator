import numpy
from wand.image import Image
from PIL import Image as PILimage
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow import strings
from tensorflow import transpose
import os


"""asciiList = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4',
             '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
             'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^',
             '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
             't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', 'Ä', 'ä', 'Ü', 'ü', 'Ö', 'ö', 'ß', "€"]"""

characters =[' ', '!', '"', '#', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5',
             '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
             'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
             'd', 'e', 'f', 'g', 'h', 'i',  'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
             'y', 'z', '\n']

characters = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
             'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
             'd', 'e', 'f', 'g', 'h', 'i',  'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
             'y', 'z']

def SplitData(path, trainevalPath="",maxlength=20):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(path)
    X = df["Tweet"].tolist()
    for i in range(len(X)):
        X[i] = str(X[i]) #fall X[i] == nan

    # Cut every Tweet at maxlength
    #for i in range(len(X)):
        #oneWord = re.search(r"([a-z]+)\s", X[i].lower())
        #if oneWord == None:
        #    print("None", end="")
        #else:
        #    X[i] = oneWord[1]
        #if len(X[i]) > maxlength:
            #X[i] = X[i][:maxlength]
    #print("")
    X_train_val, X_eval = train_test_split(X, test_size=0.2, random_state=42)

    if not trainevalPath:
        trainPath = str(path[:-4]+"_train.csv")
        evalPath = str(path[:-4]+"_eval.csv")
    else:
        trainPath = str(trainevalPath + "_train.csv")
        evalPath = str(trainevalPath + "_eval.csv")
    X_train_val = pd.DataFrame({'Tweet': X_train_val})
    X_eval = pd.DataFrame({'Tweet': X_eval})
    X_train_val.to_csv(trainPath, encoding="utf_8", index=False)
    X_eval.to_csv(evalPath, encoding="utf_8", index=False)
    return trainPath, evalPath


# Mapping characters to integers
char_to_num = layers.experimental.preprocessing.StringLookup(
    vocabulary=list(characters), num_oov_indices=0, mask_token=None
)

# Mapping integers back to original characters
num_to_char = layers.experimental.preprocessing.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


def ImageToVector(data="text-data-1", directoryname="images/", imagename="image"):
    CSV = pd.read_csv(data, encoding="utf_16")
    X = []
    y_text = CSV["Tweet"].to_list()
    for i in range(len(y_text)):
        print(i,"/",len(y_text))
        img = PILimage.open(fp=os.path.join(directoryname + imagename + str(i) + ".png"))
        x = numpy.array(img) / 255.0
        x = numpy.array(transpose(x))
        if x.shape[1] == 32:
            X += [x.tolist()]
        else:
            print("following image is not 32px tall:", directoryname +"/" + imagename + str(i) + ".png")

    y= []
    #asciiList = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    #noAscii =       ['„',   '“',  '…',  '“',    '“',    '´',    '\xa0', '\xa0', '\xa0', '²',    '´',    '„',    '“']
    #noAsciiReplace= ["\"",  "\"", ".",  "\"",   "\"",   "'",    " ",    " ",    " ",    "2",    "'",    "\"",   "\""]
    for i in range(len(y_text)):
        y += [[]]

        # Könnte sein, dass der Text NaN und dann als Float erkannt wird
        if not isinstance(y_text[i], str):
            y_text[i] = str(y_text[i])
        Tensor = numpy.array(char_to_num(strings.unicode_split(y_text[i], input_encoding="UTF-8"))) # tf.strings.unicode_split()
        y[i] = Tensor.tolist() #TODO : ???

        #for j in y_text[i]:
        #    y[i] += [characters.index(j)]

    #TODO: train test split nicht in dieser funktion machen
    #return X, y
    print("X: ",len(X),"Y: ",len(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return (X_train, y_train), (X_test, y_test)

def CheckFileExist(filename):
    import os
    try:
        with open(filename, "r", encoding="utf_16") as _:
            pass
        for _ in range(3):
            x = str("file: "+ filename+ " already exists. Do you want to replace the existing file? y/n")
            x = input(x)
            if "n" in x:
                break
            elif "y" in x:
                os.remove(filename)
                break
            else:
                print("please answer with 'y' for yes or 'n' for no and press Enter")
    except UnicodeEncodeError:
        print("There was a problem with utf_16 encoding")
    except FileNotFoundError:
        pass

def clearDirectory(path="BrandNewModel/*"):
    import glob
    import shutil

    askForRemoval = str("Do you want to clear this folder: " + path + " ?\nplease answer with y/n")
    if "y" in input(askForRemoval):
        files = glob.glob(path)
        for f in files:
            if "." in f[-4:]:
                os.remove(f)
            else:
                shutil.rmtree(f)
        print("folder is cleared")
    else:
        print("folder not cleared")
