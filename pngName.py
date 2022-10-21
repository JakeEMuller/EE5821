import os
from PIL import Image
from pytesseract import pytesseract



fileLocation = "/mnt/d/EE5821/FinalProjectData/Playground/Test_(2).png"

tesseractInstall = "/usr/share/tesseract-ocr/4.00/tessdata"

#pytesseract.tesseract_cmd = tesseractInstall

img = Image.open(fileLocation)

width, height = img.size
print(width, height)

# crop for text
leftside = 100
top = 135
rightside = (width-100)
bottom = (height-1715)
   
imgText = img.crop((leftside, top, rightside, bottom))


# crop for training data
leftside = 5
top = 130
rightside = width
bottom = (height-280)

imgData = img.crop((leftside, top, rightside, bottom))

test = pytesseract.image_to_string(imgText)



print(test)
img = imgText.save('/mnt/d/EE5821/FinalProjectData/Playground/returnName_(1).png')
img = imgData.save('/mnt/d/EE5821/FinalProjectData/Playground/returnData_(1).png')
