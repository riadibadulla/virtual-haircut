import face_recognition
import PIL.Image
import PIL.ImageDraw
import os
import PIL.ExifTags
import cv2
import math

def delete_images():
    try:
        IMAGE_DIR = "facerecognition/static/images/" + os.listdir("facerecognition/static/images/")[-1]
        os.remove(IMAGE_DIR)
    except:
        pass

def show_landmarks(pil_image,face_landmarks_list):
    # Create a PIL drawing object to be able to draw lines later
    draw = PIL.ImageDraw.Draw(pil_image)

    # Loop over each face
    for face_landmarks in face_landmarks_list:

        # Loop over each facial feature (eye, nose, mouth, lips, etc)
        for name, list_of_points in face_landmarks.items():
            # Print the location of each facial feature in this image
            print("The {} in this face has the following points: {}".format(name, list_of_points))

            # Let's trace out each facial feature in the image with a line!
            draw.line(list_of_points, fill="red", width=2)
    return pil_image

def mid_eyes(face_landmarks_list):
    right_eye_point = face_landmarks_list[0]['right_eye'][0]
    left_eye_point = face_landmarks_list[0]['left_eye'][3]
    x_dist_mid = (right_eye_point[0] - left_eye_point[0])/2
    y_dist_mid = (right_eye_point[1] - left_eye_point[1])/2
    point = (int(left_eye_point[0]+x_dist_mid),int(left_eye_point[1]+y_dist_mid))
    return point

def paste_sunglases(image,face_landmarks_list, dist):
    sunglasses = cv2.imread('facerecognition/static/haircuts/sunglasses.png', -1)
    dw = dist
    dh = int(dw * 0.50)
    glassesResize = cv2.resize(sunglasses, (dw, dh))
    glassesOriginal = glassesResize[:, :, 0:3]
    glassesOriginal = cv2.cvtColor(glassesOriginal, cv2.COLOR_BGR2RGB)
    maskGlasses = glassesResize[:, :, 3]
    mid_eye_point = mid_eyes(face_landmarks_list)

    y1 = mid_eye_point[1] - int(dh * 0.4)
    y2 = y1 + dh
    x1 = mid_eye_point[0] - int(dw / 2)
    x2 = mid_eye_point[0] + int(dw / 2)

    maskedGlassesImage = cv2.merge((maskGlasses, maskGlasses, maskGlasses))
    augGlassesMasked = cv2.bitwise_and(glassesOriginal, maskedGlassesImage)

    glassesImageROI = image.copy()
    glassesImageROI = glassesImageROI[y1:y2,x1:x2]
    glassesGirlROIImage = cv2.bitwise_and(glassesImageROI, cv2.bitwise_not(maskedGlassesImage))
    glassesGirlROIFinal = cv2.bitwise_or(glassesGirlROIImage, augGlassesMasked)
    image[y1:y2,x1:x2] = glassesGirlROIFinal
    return image

def find_distance_from_chin2chin(face_landmarks_list):
    chin = face_landmarks_list[0]['chin']
    length_of_chin = len(chin)
    point1, point2 = chin[0], chin[length_of_chin-1]
    dist = abs(point2[0]-point1[0])
    return dist

def run_landmark_detection():
    IMAGE_DIR = "facerecognition/static/images/"+os.listdir("facerecognition/static/images/")[-1]
    # IMAGE_DIR = "static/images/" + os.listdir("static/images/")[-1]

    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(IMAGE_DIR)

    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(image)

    number_of_faces = len(face_landmarks_list)
    if number_of_faces>0:
        print("I found {} face(s) in this photograph.".format(number_of_faces))
        dist = find_distance_from_chin2chin(face_landmarks_list)
        dist = int(math.ceil(dist / 100.0)) * 100

        image = cv2.imread(IMAGE_DIR)
        image = paste_sunglases(image, face_landmarks_list, dist)
        cv2.imwrite(IMAGE_DIR,image)
