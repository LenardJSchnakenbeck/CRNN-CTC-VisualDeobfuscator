from Data import Homoglyphs, Leetspeak
import random, re


def obfuscator(text, Choice):      # random: W o r t (W-o-r-t) oder \/\/oR7 oder Wort
    #re_S = re.compile(r'(\S+)')
    #tokenized = re_S.split(text)
    obf_string = []
    #for text in tokenized:
    if Choice == -1:
        obf_technique = random.randint(2, 3)  # inklusive der letzten Zahl
    else:
        obf_technique = Choice
    if obf_technique == 0:
        obf_string += [Leetobfuscator(text)]
    if obf_technique == 1:
        obf_string += [Zwischenzeichenobfuscator(text)]
    if obf_technique == 2:
        obf_string += [Einzelzeichenobfuscator(text)]
    else:
        obf_string += [text]
    return ''.join(obf_string) #repr(''.join(obf_string))

def Leetobfuscator(string, ObfuscationWahrscheinlichkeit=7):
    obfuscated_sentence = ""
    for i in string:
        choice = random.randint(1,
                                ObfuscationWahrscheinlichkeit)  # Wenn ObfuscationWahrscheinlichkeit <5 sIeHT DeR TExt So AUs
        if choice >= 7 and i.upper() in Leetspeak:
            obfuscated_sentence += str(Leetspeak[i.upper()][random.randint(0, len(Leetspeak[i.upper()]) - 1)])
        else:
            obfuscated_sentence += i.upper()  # random.choice([i.upper(),i])
    return obfuscated_sentence


def Zwischenzeichenobfuscator(string):
    obfuscated_sentence = ""
    Zwischenzeichenauswahl = " 0123456789.-_＿,｡･､ﾟ￤|￣￭￮𐊅♡:;#'+*~´`\?ß=})]([/{&%$§³\"²!"
    Zwischenzeichen = Zwischenzeichenauswahl[random.randint(0, len(Zwischenzeichenauswahl) - 1)]
    for i in string:
        if i in " ":
            obfuscated_sentence = obfuscated_sentence[:-1]
            obfuscated_sentence += i
        else:
            obfuscated_sentence += i + Zwischenzeichen
    return obfuscated_sentence


def Einzelzeichenobfuscator(text, ObfuscationWahrscheinlichkeit=4):
    Dictionary = Homoglyphs
    obfuscated_sentence = ""
    for i in text:
        randomizer = random.randint(1,ObfuscationWahrscheinlichkeit)
        if randomizer != 1 and i in Dictionary:
            obfuscated_sentence += str(
                Dictionary[i][random.randint(0, len(Dictionary[i]) - 1)])  # zufälliges Ersatzzeichen
        else:
            obfuscated_sentence += i
    return obfuscated_sentence