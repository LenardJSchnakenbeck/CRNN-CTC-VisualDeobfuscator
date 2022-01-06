import csv
import pandas as pd
import nltk
from DataManager import CheckFileExist
from unidecode import unidecode_expect_nonascii
#from text_unidecode import unidecode
from Levenshtein import distance, editops
import pytesseract
from ImageRender import csv_obf_img, createImage
import os
import re
import matplotlib


def CreateEvaluationFile(present_data="BrandNewModel/dataset_eval.csv", evaluationFile="BrandNewModel/evaluation.csv",
                         imagepath="data/evalimages", imagename="evalimage"):
    CheckFileExist(evaluationFile)
    csv_obf_img(imagename=imagename, filename=evaluationFile, data=present_data, imagepath=imagepath)
    """present_data = pd.read_csv(present_data, encoding="utf_16")
    j=0
    obfTweets=[]
    for i in present_data["Tweet"]:
        print(i,"/",len(present_data["Tweet"]))
        obfuscated_Satz = i #present_data["obfuscated_Tweet"][i]
        createImage(obfuscated_Satz, j, name=imagename, path=imagepath)
        j+=1
    present_data.to_csv(evaluationFile, encoding="utf_16", index=False)"""

def levenshteinAnalysis(deobfuscatedTweet, tweet):
    # returns levenshteindistance & count of the 3 edit operations listed below
    deletions = []
    insertions = []
    replacements = []
    deobfuscatedTweet = deobfuscatedTweet.lower()
    tweet = tweet.lower()
    edits = editops(deobfuscatedTweet, tweet.lower())
    for i in edits:
        if i[0] == "delete":
            deletions += deobfuscatedTweet[i[1]]
        elif i[0] == "insert":
            insertions += tweet.lower()[i[2]]
        elif i[0] == "replace":
            replacements += [[deobfuscatedTweet[i[1]], tweet[i[2]]]]
    dist = distance(deobfuscatedTweet, tweet) / max(len(deobfuscatedTweet), len(tweet)) #normalized
    return (dist, len(deletions), len(insertions), len(replacements)), (deletions, insertions, replacements), edits

def getTotalMeasurements(data="BrandNewModel/evaluation.csv",column="CRNNLevenshtein"):
    CSV = pd.read_csv(data, encoding="utf_16")
    deletions = 0
    replacements = 0
    insertions = 0
    distances = 0
    for edits in CSV[column]:
        edits = re.match(r"(\(\s?(\d+\.?\d+),\s?(\d+),\s?(\d+),\s?(\d+)\))", edits)
        distances += float(edits[2])
        deletions += int(edits[3])
        insertions += int(edits[4])
        replacements += int(edits[5])

    return (distances/len(CSV[column]), deletions, insertions, replacements)


#====================== Models ========================================================
pathToModel = r"/Users/lenard/Downloads/BrandNewModel"
data = r"/Users/lenard/Downloads/BrandNewModel/dataset_eval.csv"
imagepath = "/Users/lenard/Downloads/images/"
imagename = "image"
deobfMethod = "CRNN"

from PIL import Image as PILimage
import numpy as np
from DataManager import num_to_char
from tensorflow import keras, transpose, strings

def decode_batch_predictions(pred):
    max_length = 25 #Length of longest word to predict
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
              # TODO: greedy / beamsearch
              :, :max_length
              ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

def img_to_string_CRNN(imagepath, pathToModel=None): #"/Users/lenard/Downloads/BrandNewModel/"
    if 'model' not in globals():
        global model
        #from FinalModelCRNN import decode_batch_predictions  # still takes some time
        from tensorflow import keras
        from keras import layers

        if pathToModel == None: print("pls give pathToModel as secound argument to this function")
        print("load deobfuscator...")
        model = keras.models.load_model(pathToModel)
        model = keras.models.Model(
            model.get_layer(name="image").input, model.get_layer(name="dense2").output
        )

    def img_to_vector(imagepath):
        img = PILimage.open(imagepath)
        vector = np.array(img) / 255.0
        vector = np.array(transpose(vector))
        vector = vector.tolist()
        return vector

    def predict_from_vector(imagepath):
        input = img_to_vector(imagepath=imagepath)
        prediction = model.predict(np.array([input, ]))
        prediction = decode_batch_predictions(prediction)
        return prediction
    deobfTweet = predict_from_vector((imagepath))
    return deobfTweet[0].replace("[UNK]", "") #.lower()

#deobfuscateAndLevenshtein("CRNN", "/Users/lenard/Downloads/BrandNewModel/", "/Users/lenard/Downloads/BrandNewModel/evaluation.csv", "image", "/Users/lenard/Downloads/evalimages/")
def deobfuscateAndLevenshtein(deofMethod, pathToModel=None, data="data/evaluation.csv", imagename="evalimage",
                imagepath="data/evalimages/"):  # pathToModel = "bittrashModel-6characters/originalModel"
    pytesseract.pytesseract.tesseract_cmd = "home/ba/miniconda3/envs/deobf/bin/tesseract.exe" # "/usr/local/Cellar/tesseract/4.1.3/bin/tesseract.exe"

    CSV = pd.read_csv(data, encoding="utf_16")
    Deobfuscated = []
    Levenshtein = []

    # total numbers, over all tweets
    deletions = 0
    replacements = 0
    insertions = 0
    distances = 0
    deletedChars = []
    insertedChars = []
    replacedChars = []

    allEditops = []

    j = 0
    print(len(CSV["Tweet"]), " Tweets get deobfuscated...")
    for tweet in CSV["Tweet"]:
        if j%100 == 0: print(j,"/",len(CSV["Tweet"]))

        if deofMethod == "CRNN":
            csvColumnDeobf = "CRNNDeobf"
            csvColumnLevenshtein = "CRNNLevenshtein"

            image = os.path.join(imagepath, str(imagename + str(j) + ".png") )
            deobfTweet = img_to_string_CRNN(image, pathToModel)[0].lower()
            try:
                deobfTweet = img_to_string_CRNN(image, pathToModel)[0].lower()     # .strip()
            except:
                deobfTweet = ""
                print("Deobfuscating image number % eraised an Error, maybe because it has a very small width?" % j)
            #deobfTweet = deobfTweet.replace("[unk]","") #□                          # "UNK".lower() --> "unk"

        elif deofMethod == "pytesseract":
            csvColumnDeobf="pytesseractDeobf"
            csvColumnLevenshtein="pytesseractLevenshtein"
            deobfTweet = pytesseract.image_to_string(os.path.join(imagepath + imagename + str(j) + ".png"),
                                                   lang="eng").strip()
            if "\n" in deobfTweet:
                deobfTweet = deobfTweet.replace("\n", " ") #TODO: maybe we need to strip harder

        elif deofMethod == "normalizer":
            csvColumnDeobf = "normalizerDeobf"
            csvColumnLevenshtein = "normalizerLevenshtein"

            obfTweet = CSV["obfuscated_Tweet"][j]
            deobfTweet = unidecode_expect_nonascii(obfTweet)  # unidecode(obfTweet) #
            deobfTweet = deobfTweet.replace("[?]", "")
        else:
            print("which deobfMethod do you want to use? deobfuscator/pytesseract/normalizer")
            return


        edits, wrongChars, rawedits = levenshteinAnalysis(deobfTweet, tweet)
        distances += edits[0]
        deletions += edits[1]
        insertions += edits[2]
        replacements += edits[3]
        deletedChars += [wrongChars[0]]
        insertedChars += [wrongChars[1]]
        replacedChars += [wrongChars[2]]
        allEditops += [rawedits]

        Levenshtein += [edits]
        Deobfuscated += [deobfTweet]
        j += 1

    CSV[csvColumnDeobf] = Deobfuscated
    CSV[csvColumnLevenshtein] = Levenshtein
    CSV.to_csv(data, encoding="utf_16", index=False)
    print(deofMethod, "/n", "dis, del, ins, rep: ",
          (distances / len(CSV[csvColumnDeobf]), deletions, insertions, replacements))
    return (distances, deletions, insertions, replacements), (deletedChars, insertedChars, replacedChars), allEditops


def applyCompareModels(
    imagepath = "C:/Users/Lenard/Downloads/export(3)/BrandNewModel/evalimagesIMGspace/",
    imagename = "evalimage",
    evaluationFile = r"C:\Users\Lenard\Downloads\export(3)\BrandNewModel\evaluationIMGspace.csv",
    evaldata = r"C:\Users\Lenard\Downloads\export(3)\BrandNewModel\dataset_eval.csv"):

    # this obfuscates the evaluationdata and creates images
    # CreateEvaluationFile(present_data=evaldata, evaluationFile=evaluationFile,
    #                     imagepath=imagepath, imagename=imagename)

    pathToModel = r"C:\Users\Lenard\Downloads\export(3)\BrandNewModel"  # "BrandNewModel/"
    df = pd.read_csv(evaluationFile, encoding="utf_16")
    j = 0
    for i in df["obfuscated_Tweet"]:
        createImage(i, j, imagename, imagepath)
        j += 1

    totalEditsCRNN, wrongCharsCRNN, editsCRNN = deobfuscateAndLevenshtein("CRNN",pathToModel, data=evaluationFile, imagename=imagename,
                                                         imagepath=imagepath)
    totalEditsNorm, wrongCharsNorm, editsNorm = deobfuscateAndLevenshtein("normalizer",data=evaluationFile)
    totalEditsPyTess, wrongCharsPyTess, editsPyTess = deobfuscateAndLevenshtein("pytesseract",data=evaluationFile, imagename=imagename,
                                                                     imagepath=imagepath)

    newModelMeasurements = getTotalMeasurements(
        data=evaluationFile, column="CRNNLevenshtein")
    normalizerMeasurements = getTotalMeasurements(
        data=evaluationFile, column="normalizerLevenshtein")
    pyTessMeasurements = getTotalMeasurements(
        data=evaluationFile, column="pytesseractLevenshtein")

    charCount = 0
    df = pd.read_csv(evaluationFile, encoding="utf_16")
    pytessDist = df["pytesseractLevenshtein"]
    for i in df["obfuscated_Tweet"]:
        charCount += len(i)

    # print(newModelDist, normalizerDist)#, sum(pytessDist))
    # print(newModelDist/charCount) #charCount is smaller then total Levenshtein distance ???

########################################################################################################################
####################################### E V A L U A T I O N ############################################################
########################################################################################################################

# Replacement Character Frequency
def RepCharsFreq(Replacements): # = wrongCharsOCR[2]
    wrongies = []
    for i in Replacements:
        if i != []:
            wrongies += [i]
    for i in range(len(wrongies)):
        a = ""
        for k in range(len(wrongies[i])):
            for j in wrongies[i][k]:
                a += j
        wrongies[i] = a
    reps = wrongies
    stopp = True
    while stopp:
        stopp = False
        for i in range(len(reps)):
            if len(reps[i]) > 2:
                stopp = True
                reps += [reps[i][-2:]]
                reps[i] = reps[i][:-2]
    freq = nltk.FreqDist(char for char in reps) #char[1] for char -> to compare single chars

    import matplotlib.pyplot as plt;
    plt.rcdefaults()
    import numpy as np
    import matplotlib.pyplot as plt
    chars = [i[0] for i in freq.most_common(25)]
    y_pos = np.arange(len(chars))
    count = [i[1] for i in freq.most_common(25)]
    plt.bar(y_pos, count, align='center', alpha=0.5)
    plt.xticks(y_pos, chars)
    plt.ylabel('count of replacements')
    plt.xlabel('predicted character, target character')
    plt.title('Frequence Of Character Replacements')
    return freq

def CharDistribution():
    from DataManager import characters
    charDist = {}
    for i in characters:
        if i == i.lower():
            charDist[i] = 0
    CSV = pd.read_csv(evaluationFile, encoding="utf_16")
    for i in CSV["Tweet"]:
        for j in i:
            charDist[j.lower()] += 1
    freq = nltk.FreqDist(charDist)
    return freq

# ? Edit-Operations per Model ?
def plotOperations(normalizerMeasurements, pyTessMeasurements, newModelMeasurements, oldModelMeasurements):
    #evaluationFile = r"C:\Users\Lenard\Downloads\export(3)\BrandNewModel\evaluationN-oldImageRender.csv"
    #oldModelMeasurements = getTotalMeasurements(data=evaluationFile, column="CRNNLevenshtein")
    #evaluationFile = r"C:\Users\Lenard\Downloads\export(3)\BrandNewModel\evaluationN.csv"
    #newModelMeasurements = getTotalMeasurements(data=evaluationFile, column="CRNNLevenshtein")
    #normalizerMeasurements = getTotalMeasurements(data=evaluationFile,
    #                                              column="normalizerLevenshtein")  # dist del ins rep
    #pyTessMeasurements = getTotalMeasurements(data=evaluationFile, column="pytesseractLevenshtein")

    import matplotlib.pyplot as plt
    import numpy as np

    labels = ['Normalizer', 'PyTesseract OCR', 'proposed Model']#, 'proposed Model\n(images with space)']
    deletions = [normalizerMeasurements[1], pyTessMeasurements[1], newModelMeasurements[1]]#oldModelMeasurements[1]]#, 21]
    insertions = [normalizerMeasurements[2], pyTessMeasurements[2], newModelMeasurements[2]]#oldModelMeasurements[2]]#,
    replacements = [normalizerMeasurements[3], pyTessMeasurements[3], newModelMeasurements[3]]#oldModelMeasurements[3]]#,

    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots()
    rects0 = ax.bar(x, insertions, width, label='Insertions')
    rects1 = ax.bar(x, replacements, width, bottom=insertions, label='Replacements')
    rects2 = ax.bar(x, deletions, width, bottom=[insertions[i]+replacements[i] for i in range(len(insertions))],
                    label='Deletions')
    #rects3 = ax.bar(x, [0,0,0,newModelMeasurements[1]-21], width, bottom=[insertions[i]+replacements[i]+deletions[i] for i in range(len(insertions))],
    #                label='Deletions\ncaused by whitespace')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Count of Edit Operations')
    ax.set_title('Operations by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    plt.show()

def wordLength(length): # length = freDist of form: {6: 2442, 7: 2405, 5: 2179 ...}
    import matplotlib.pyplot as plt
    plt.rcdefaults()
    import numpy as np
    import matplotlib.pyplot as plt
    chars = [i for i in range(len(length))]
    y_pos = np.arange(len(chars))
    count = [length[i] for i in chars]
    plt.bar(y_pos, count, align='center', alpha=0.5)
    plt.xticks(y_pos, chars)
    plt.ylabel('count of words with length x')
    plt.xlabel('word length')
    plt.title('word length distribution')

################################# H A T E - S P E E C H ################################################################

def hatespeechClassifier(tweets): #["We are very happy to show you the 🤗 Transformers library.", "I hate those terrorist wogs."]
    #from transformers import TextClassificationPipeline

    #classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    #inputs = tokenizer("I hate those terrorist wogs")
    pt_batch = tokenizer(
        tweets,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    #for key, value in pt_batch.items():
    #    print(f"{key}: {value.numpy().tolist()}")
    pt_outputs = model(**pt_batch)
    pt_predictions = F.softmax(pt_outputs[0], dim=-1)
    #print(pt_predictions)
    return pt_predictions

def interpretHSclass(tensor):
    highestclass = []
    for i in tensor:
        pred = max(i)
        if i[0] == pred:
            highestclass += ["hate"]
        elif i[1] == pred:
            highestclass += ["normal"]
        elif i[2] == pred:
            highestclass += ["offensive"]
    return highestclass

def hatespeechEval(): #evalfile = "BrandNewModel/evaluationHateSpeech.csv"):
    from itertools import product
    import pandas as pd
    import os
    import pickle
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch.nn.functional as F
    tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
    model = AutoModelForSequenceClassification.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")

    try:
        os.mkdir("BrandNewModel/HSimages/")
    except:
        pass

    imagepath = "BrandNewModel/HSimages/"
    imagename = "s7quiggle"
    evalfile = "BrandNewModel/s7quiggleEval.csv" #mitliefern
    pathToModel = "BrandNewModel/" #mitliefern


    #imagepath = "C:/Users/Lenard/Downloads/export(3)/BrandNewModel/evalimagesHateSpeech/"
    #imagename = "evalimageHS"
    #evalfile = r"C:\Users\Lenard\Downloads\export(3)\BrandNewModel\evaluationHateSpeech.csv"
    pathToModel = r"C:\Users\Lenard\Downloads\export(3)\BrandNewModel"

    profanities = ['gay', 'gays', 'lesbian', 'fag', 'faggot', 'faggots', 'faggotclub', 'queer', 'fuck', 'fucking',
                   'fuckin', 'cocksucker', 'retard', 'idiot', 'moron', 'dumbass', 'stupid', 'incompetent', 'delusional',
                   'douchebag', 'fucktard', 'dumbfuck', 'stupidtrump', 'cunt', 'cunts', 'bitch', 'bitches',
                   'bitching', 'pussy', 'dick', 'dicks', 'cock', 'dogs', 'dog', 'bull', 'islam', 'islamic', 'jesus',
                   'god', 'devil', 'hell', 'godking', 'nigger', 'nigga', 'niggas', 'niggers', 'sandnigger', 'beaner',
                   'wogs', 'racist', 'racists', 'islamophobia', 'rapist', 'pissedr', 'bastard', 'bastards', 'sucker',
                   'rap', 'bullcrap', 'piece', 'shit', 'shithead', 'damn', 'damnit', 'fucker',
                   'motherfucker', 'motherfucking']

    sents = ["my neighbor is a ", "happy birthday, ", "I hate ", "go away, you ", "do you know a ", "it is rude to say ",
             "I like ", "we love ", "unfortunately, you are a ", "there are humans and ", "never call someone a ",
             "burning, should all ", "my boss is a ", "nobody likes ", "nowadays, here are many ", "flowers are like ",
             "Who sees ", "It sounds like ", "I need to see a "]

    sents = ["my neighbor is a ", "my boss is a ", "you are a ",
             "nobody likes ", "I hate ", "It makes me gag to think about ", "I would kill myself, if I would be a ",
             "It should be allowed to kill a ", "don't believe the lies of this ", "never trust a ",
             "my daughter is not allow to marry this ", "there are humans and ", "nowadays, here are too many ", "go away, you ",
             "why should someone choose being a ", "this is full of ", "Yesterday, I met a ",

             "happy birthday, ", "do you know a ", "It is okay to be a", "You should be proud to be a",
             "it is rude to say ", "never call someone a "
             ]

    a = [sents, profanities]
    dataset = list(product(*a))
    datasetonlyprof = [i[1] for i in dataset]
    datasetonlysent = [i[0] for i in dataset]

    def createData():
        CreateEvaluationFile(present_data=datasetonlyprof,
                             evaluationFile=evalfile,
                             imagepath=imagepath, imagename=imagename)
    createData()
    CSV = pd.read_csv(evalfile, encoding="utf_16")

    # Tweet = profanity

    #for i in range(len(df["obfuscated_Tweet"])):
    #    createImage(df["obfuscated_Tweet"][i], i, imagename, imagepath)

    CSV["sentences"] = datasetonlysent
    totalEditsCRNN, wrongCharsCRNN, error = deobfuscateAndLevenshtein("CRNN",pathToModel, data=evalfile, imagename=imagename,
                                                      imagepath=imagepath)
    newModelMeasurements = getTotalMeasurements(data=evalfile, column="CRNNLevenshtein")

    print(CSV.keys())
    CSV = pd.read_csv(evalfile, encoding="utf_16")

    classification = [str(datasetonlysent[i]+CSV["Tweet"][i]) for i in range(len(datasetonlysent))]
    classificationObf = [str(datasetonlysent[i]+CSV["obfuscated_Tweet"][i]) for i in range(len(datasetonlysent))]
    classificationDeobf = [str(datasetonlysent[i]+CSV["CRNNDeobf"][i]) for i in range(len(datasetonlysent))]

    classifiedAll = [] #unobf, obf, deobf

    for i in classification:  #raw profanity
        classified1 = hatespeechClassifier(i)
        classified2 = interpretHSclass(classified1)
        classifiedAll.append([*classified1[0], classified2]) #*classfication -> tupel unpacking
    CSV["classification"] = classifiedAll
    with open('BrandNewModel/classraw.pkl', 'wb') as f:
        pickle.dump(classifiedAll, f)
    classraw = classifiedAll

    CSV.to_csv(evalfile, encoding="utf_16")

    classifiedAll = []
    for i in classificationObf:  #obfuscated profanity
        classified1 = hatespeechClassifier(i)
        classified2 = interpretHSclass(classified1)
        classifiedAll.append([*classified1[0], classified2])  # *classfication -> tupel unpacking
    CSV["classificationObf"] = classifiedAll
    with open('BrandNewModel/classObf.pkl', 'wb') as f:
        pickle.dump(classifiedAll, f)

    classifiedAll = []
    for i in classificationDeobf:  #deobfuscated profanity
        classified1 = hatespeechClassifier(i)
        classified2 = interpretHSclass(classified1)
        classifiedAll.append([*classified1[0], classified2])
    CSV["classificationDeobf"] = classifiedAll
    with open('BrandNewModel/classDeobf-knownobf.pkl', 'wb') as f:
        pickle.dump(classifiedAll, f)

    CSV.to_csv(evalfile, encoding="utf_16")
    with open('BrandNewModel/unknownobf-editcount-wrongchars-error-edits.pkl', 'wb') as f:
        pickle.dump([totalEditsCRNN, wrongCharsCRNN, error], f)

    # this produces a list of the classification before obfuscating, after and after deobfuscating.
    beforeafter = [str(classraw[i][-1][0][0] + classObf[i][-1][0][0] + classDeobf[i][-1][0][0]) for i in
                   range(len(classraw))]

    b = {}
    for i in sents:
        b[i] = []
    j = 0
    for i in [classraw[i][-1][0][0] for i in range(len(classraw))]:
        b[sents[j // 69]] += i
        j += 1
    for i in b.keys():
        temp = nltk.FreqDist(b[i])
        b[i] = [temp["n"], temp["o"], temp["h"]]

    #Average:
    for i in sents:
        for j in range(3):
            blist[j] += int(b[i][j])
    temp = sum(blist)
    for i in range(3):
        blist[i] = blist[i] / temp


    a = {}
    for i in profanities:
        a[i] = []

    j = 0
    for i in [classraw[i][-1][0][0] for i in range(len(classraw))]: #[i for i in classraw]:
        a[profanities[j]] += i
        j += 1
        if j == 67:
            j = 0

    import matplotlib.pyplot as plt

    # Create a array of marks in different subjects scored by different students
    marks = [a[i] for i in a.keys()]

    for i in range(len(marks)):
        for j in range(len(marks[i])):
            marks[i][j] = "noh".index(marks[i][j])

    # name of students
    names = sents
    # name of subjects
    subjects = profanities

    # Setting the labels of x axis.
    # set the xticks as student-names
    # rotate the labels by 90 degree to fit the names
    plt.xticks(ticks=np.arange(len(names)), labels=names, rotation=90)
    # Setting the labels of y axis.
    # set the xticks as subject-names
    plt.yticks(ticks=np.arange(len(subjects)), labels=subjects)
    # use the imshow function to generate a heatmap
    # cmap parameter gives color to the graph
    # setting the interpolation will lead to different types of graphs
    plt.imshow(marks, cmap='cool', interpolation="nearest")
    matplotlib
    heatmap
