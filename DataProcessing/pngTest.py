import os
import csv
import cv2
import numpy as np
from PIL import Image
from pytesseract import pytesseract



def main():
    #tesseractInstall = "/usr/share/tesseract-ocr/4.00/tessdata"

    # angles to get data from
    anglesToConvert = ["25","30","35","40","45","50"]
    #anglesToConvert = ["50"]
    dir = "/mnt/d/EE5821/FinalProjectData/LD_DATA/"


    # open new CSV for data

    #get directory template
    #dir_path = "/mnt/d/EE5821/FinalProjectData/LD_DATA/25Degrees-Raw/"
    
    #dir_path = "/mnt/d/EE5821/FinalProjectData/LD_DATA/playground/"

    # -------------------------------
    # set up if video template
    # -------------------------------
    videoImgDir = "/mnt/d/EE5821/FinalProjectData/LD_DATA/remove/SampleVideo.png"
    videoTemplate = Image.open(videoImgDir)
    width, height = videoTemplate.size

    leftside = 5
    top = height - 265
    rightside = width
    bottom = (height-200)
    videoTemplate = videoTemplate.crop((leftside, top, rightside, bottom))
    videoTemplate = np.array(videoTemplate)
    # Convert RGB to BGR 
    videoTemplate = videoTemplate[:, :, ::-1].copy() 

    # -------------------------------
    # set up if video template
    # -------------------------------
    MatchImgDir = "/mnt/d/EE5821/FinalProjectData/LD_DATA/remove/SampleMatch.png"
    MatchTemplate = Image.open(MatchImgDir)
    width, height = MatchTemplate.size

    leftside = 459
    top = 207
    rightside = 482
    bottom = (height-1690)
    MatchTemplate = MatchTemplate.crop((leftside, top, rightside, bottom))
    MatchTemplate = np.array(MatchTemplate)
    # Convert RGB to BGR 
    MatchTemplate = MatchTemplate[:, :, ::-1].copy() 

    for i in anglesToConvert:
        performAngle(dir, i, videoTemplate, MatchTemplate)
    
    print("done")

# ------------------------------------------
# Formats data for a specific angle 
# ------------------------------------------
def performAngle(dir, angle, videoTemplate, MatchTemplate):
    f = open(dir + 'DataTemplate' + angle + 'Degrees.csv', 'w+')
    f.write('File Name,Route Name,Rating,Angle,NoMatching\n')

    dir_data = dir + angle + 'Degrees_Raw/'
    dir_result = dir + angle + 'Degrees_Cropped/'
    picNum = len([entry for entry in os.listdir(dir_data) if os.path.isfile(os.path.join(dir_data, entry))])
    #picNum,= 5
    for i in range(1,(picNum+1)):

        if((i % 50) == 0):
            print("Angle " + angle + ', Image ' + str(i))

        fileLocation = dir_data + 'Data-(' + str(i) + ').png'
        img = Image.open(fileLocation)
        width, height = img.size
        #print(width, height)       
    
        ##check if video
        if( not isVideo(img,videoTemplate)):

            noMatch = 0
            if(isNoMatch(img, MatchTemplate)):
                noMatch = 1
            # crop for training data
            leftside = 5
            top = 450
            rightside = width
            bottom = (height-300)
    
            imgData = img.crop((leftside, top, rightside, bottom))

            title, rating = getNameAndDifficulty(img, width, height)
            #angle = getAngle(img, width, height)
    
            f.write('CropData-(' + str(i) + '),' + title + ',' + rating + "," + angle + "," + str(noMatch) +'\n')
            #img = imgAngle.save('/mnt/d/EE5821/FinalProjectData/LD_DATA/Test/CropDataAngle-(' + str(i) + ').png')
            #img = imgData.save('/mnt/d/EE5821/FinalProjectData/LD_DATA/Test/CropDataData-(' + str(i) + ').png')
            #img = imgText.save('/mnt/d/EE5821/FinalProjectData/LD_DATA/Test/CropDataText-(' + str(i) + ').png')
            #img = imgData.save('/mnt/d/EE5821/FinalProjectData/LD_DATA/25Degrees-Cropped/CropData-(' + str(i) + ').png')
            img = imgData.save( dir_result + 'CropData-(' + str(i) + ').png')

    
    f.close()
    return 

# -----------------------------------------
# Checks if data has no match
# -----------------------------------------
def isNoMatch(image, templateCV2):
    method = cv2.TM_CCOEFF_NORMED
    # convert to cv2 format
    ImgCV2 = np.array(image)
    # convert to BGR
    ImgCV2 = ImgCV2[:, :, ::-1].copy() 

    heatmap = cv2.matchTemplate(ImgCV2, templateCV2, method)
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    #print('Heatmap cords')
    #print(y,x)
    if(y == 207):
        #print("Was Match")
        return 1
    else:
        return 0

# -----------------------------------------
# Dev tool to determine cropping for nomatch
# -----------------------------------------
def NoMatchTest():
    
    MatchImgDir = "/mnt/d/EE5821/FinalProjectData/LD_DATA/remove/SampleMatch.png"
    dir_result = "/mnt/d/EE5821/FinalProjectData/LD_DATA/remove/"
    MatchTemplate = Image.open(MatchImgDir)
    width, height = MatchTemplate.size

    leftside = 459
    top = 207
    rightside = 482
    bottom = (height-1690)
    MatchTemplate = MatchTemplate.crop((leftside, top, rightside, bottom))
    

    img = MatchTemplate.save( dir_result + 'Test-(' + "1" + ').png')

    MatchTemplate = np.array(MatchTemplate)
    # Convert RGB to BGR 
    MatchTemplate = MatchTemplate[:, :, ::-1].copy() 

    print(isNoMatch(Image.open(MatchImgDir), MatchTemplate))

    return 0
# -----------------------------------------
# Checks if data is a video route
# -----------------------------------------
def isVideo(image, templateCV2):
    method = cv2.TM_CCOEFF_NORMED
    # convert to cv2 format
    ImgCV2 = np.array(image)
    # convert to BGR
    ImgCV2 = ImgCV2[:, :, ::-1].copy() 

    heatmap = cv2.matchTemplate(ImgCV2, templateCV2, method)
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    #print('Heatmap cords')
    #print(y,x)
    if(y == 1655 and x == 5):
        #print("Was Video")
        return 1
    else:
        return 0

# -----------------------------------------
# Gets name and dificulty from image
# -----------------------------------------
def getNameAndDifficulty(Image, width, height):
    # crop for text
    leftside = 100
    top = 155
    rightside = (width-100)
    bottom = (height-1665)
    
    imgText = Image.crop((leftside, top, rightside, bottom))
    
    ImageText = pytesseract.image_to_string(imgText)
    listOfText = ImageText.split('\n')
    #print(listOfText)

    ratingIndex = listOfText[1].find('V')
    if ratingIndex == -1:
        ratingIndex = listOfText[1].find('v')
    
    title = listOfText[0]
    title = title.replace(",", "-")
    rating = listOfText[1][ratingIndex+1]
    rating = rating.replace("o","0")
    rating = rating.replace("O","0")


    if ratingIndex == -1:
        print('ERROR')
        rating = 'ERROR'


    return title, rating;

# -----------------------------------------
# Gets angle from image (Does not work)
# -----------------------------------------
def getAngle(Image, width, height):
    
    # crop for angle
    leftside = 600
    top = 50
    rightside = (width-375)
    bottom = (height-1770)
    imgAngle = Image.crop((leftside, top, rightside, bottom))

    ImageText = pytesseract.image_to_string(imgAngle)
    listOfTextAngle = ImageText.split('\n')
    angle = listOfTextAngle[0][0] + listOfTextAngle[0][1]; 

    #tesseract doesnt like 20 degrees for some reason
    if angle == "yA":
        angle = "20"
    if angle == "Ki":
        angle = "30"
    if angle == "ci":
        angle = "40"
    if angle == "ce":
        angle = "45"



    #print(angle)
    return angle

#call main
main()
#NoMatchTest()
