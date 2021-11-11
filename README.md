Unicode Homograph Deobfuscator using an OCR CRNN trained with obfuscated online posts


Basically a Tensorflow 2.x implementation of https://github.com/Belval/CRNN, which is a Tensorflow 1.x implementation of https://github.com/bgshih/crnn.
Described in the following paper by Shi et al http://arxiv.org/abs/1507.05717.

# This is an example - Ťɧϊṣ іŝ ἇἡ eẋa₥ҏℓē
Further the Code contains an Obfuscator, that changes clear text 
(from the datasets HASOC (https://hasocfire.github.io/hasoc/2019/dataset.html) and CONAN (https://github.com/marcoguerini/CONAN)) 
to obfuscated text, by substitution of single characters with homographs from the unicode.

The obfuscated posts from the datasets get rendered to images with a height of 32. (The width depends on the text.) The vectorized Images form the input to the convolutional recurrent neural network (CRNN) and the clear text is used as goldlabel.
Thus the model learns to deobfuscate strings by their optical appearence.

Additionally, the performance of the model and two baselines (pytesseract & unidecode-normalizer) can be evaluated using Levenshtein, as well as an hate speech detector.
The difference between the outputs of hate speech detection for the original text, obfuscated text and deobfuscated text get measured.

# Set Up
clone git
install ImageMick and pytesseract 4.x
install python 3.8 and the requirements
download unifont from http://unifoundry.com/pub/unifont/unifont-13.0.05/font-builds/unifont-13.0.05.ttf
   insert its full path into: ImageRender.py > createImage() > font = r"C:\..."
change the path for the tesseract.exe in Evaluation.py > deobfuscateAndLevenshtein() > pytesseract.pytesseract.tesseract_cmd = r"C:\..."
create folder for the images (/data/images by default)
create folder to save the model and its train history (/BrandNewModel by default, used in FinalModelCRNN.py)

```bash
apt-get update && apt-get install git wget fontconfig libmagickwand-dev
mkdir -p ~/.fonts
wget -O ~/.fonts/unifont-13.0.05.ttf http://unifoundry.com/pub/unifont/unifont-13.0.05/font-builds/unifont-13.0.05.ttf
fc-cache -fv
git clone https://github.com/LenardJSchnakenbeck/deobfuscator.git
cd deobfuscator
mkdir -p data/images
mkdir BrandNewModel

export CUDA_VISIBLE_DEVICES=2
```
