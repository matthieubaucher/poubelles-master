# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 11:01:53 2021

@author: Alexandre.Iborra
"""

def DetectBlurFaceOld():
    
    # Reading an image using OpenCV
    # OpenCV reads images by default in BGR format
    image = cv2.imread("C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/application/data/demonstration/dechet_detect/predicted_img.png")
      
    # Converting BGR image into a RGB image
    picture = cv2.imread("C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/application/data/demonstration/dechet_detect/predicted_img.png")
    #transfo = np.ones(picture.shape, dtype="uint8")*100
    gray_picture = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    #rgb2bgr = cv2.cvtColor (picture, cv2.COLOR_RGB2BGR)
    #lighten = cv2.add(picture, transfo)

    # Load the Cascade Manual Classifier
    harr_cascade = cv2.CascadeClassifier("../config/haarcascade_frontalface_default.xml")

    face_coords = harr_cascade.detectMultiScale(gray_picture, scaleFactor=1.13, minNeighbors=3) # minNeighbours @ ~>1 => better face detection also FP rate increased
    
    for x, y, w, h in face_coords:
        blur_face = picture[y:y+h, x:x+w]
        blur_face = cv2.GaussianBlur(blur_face,(351, 151), 0) # Privacy control Strength
        picture[y:y+blur_face.shape[0], x:x+blur_face.shape[1]] = blur_face
        
    numb = ""
    cv2.imwrite('../data/demonstration/dechet_detect/predicted_img.png', picture)

def DetectBlurFace():
    
    #image = cv2.imread("data/demonstration/dechet_detect/predicted_img.png")
    image = cv2.imread("C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/application/data/demonstration/dechet_detect/predicted_img.png")
    prototxt_path = "C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/application/config/deploy.prototxt"
    model_path = "C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/application/config/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    # get width and height of the image
    h, w = image.shape[:2]
    # gaussian blur kernel size depends on width and height of original image
    kernel_width = (w // 7) | 1
    kernel_height = (h // 7) | 1
    
    # preprocess the image: resize and performs mean subtraction
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    # set the image into the input of the neural network
    model.setInput(blob)
    # perform inference and get the result
    output = np.squeeze(model.forward())

    for i in range(0, output.shape[0]):
        confidence = output[i, 2]
        # get the confidence
        # if confidence is above 40%, then blur the bounding box (face)
        if confidence > 0.15:
            # get the surrounding box cordinates and upscale them to original image
            box = output[i, 3:7] * np.array([w, h, w, h])
            # convert to integers
            start_x, start_y, end_x, end_y = box.astype(int)
            # get the face image
            face = image[start_y: end_y, start_x: end_x]
            # apply gaussian blur to this face
            face = cv2.GaussianBlur(face, (kernel_width, kernel_height), 0)
            # put the blurred face into the original image
            image[start_y: end_y, start_x: end_x] = face
            
            # cv2.imwrite('C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/application/data/demonstration/images_floutages/results2/gopro478_43.png', image)
            cv2.imwrite('C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/application/data/demonstration/dechet_detect/predicted_img.png', image)

def DetectBlurFace3():
    
    #picture = image_vid
    image = cv2.imread("C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/application/data/demonstration/dechet_detect/predicted_img.png")

    gray_picture = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    #harr_cascade = cv2.CascadeClassifier("data/demonstration/blur/harr_face_detect_classifier.xml")
    harr_cascade = cv2.CascadeClassifier("data/demonstration/blur/haarcascade_frontalface_default.xml")
    harr_cascade = cv2.CascadeClassifier("../config/model_floutage.xml")

    #face_coords = harr_cascade.detectMultiScale(gray_picture, scaleFactor=1.08, minNeighbors=3) # minNeighbours @ 1 => better face detection + FP rate increased
    face_coords = harr_cascade.detectMultiScale(gray_picture, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30),  maxSize=(120, 120))
    #     blur_face
    for x, y, w, h in face_coords:
        blur_face = image[y:y+h, x:x+w]
        blur_face = cv2.GaussianBlur(blur_face,(351, 151), 0) # Privacy control Strength
        image[y:y+blur_face.shape[0], x:x+blur_face.shape[1]] = blur_face 
    
    cv2.imwrite('C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/application/data/demonstration/dechet_detect/predicted_img.png', image)
        
# Test du floutage

#image_floutage = load_img("C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/application/data/demonstration/images_floutages/gopro478_30.PNG")

#DetectBlurFace3(np.float32(image_floutage))

