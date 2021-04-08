import face_recognition
import PIL.Image
import PIL.ImageDraw
import os
import PIL.ExifTags

def delete_images():
    try:
        IMAGE_DIR = "facerecognition/static/images/" + os.listdir("facerecognition/static/images/")[-1]
        os.remove(IMAGE_DIR)
    except:
        pass


def run_landmark_detection():
    IMAGE_DIR = "facerecognition/static/images/"+os.listdir("facerecognition/static/images/")[-1]


    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(IMAGE_DIR)

    try:
        if hasattr(image, '_getexif'):  # only present in JPEGs
            for orientation in PIL.ExifTags.TAGS.keys():
                if PIL.ExifTags.TAGS[orientation] == 'Orientation':
                    break
            e = image._getexif()  # returns None if no EXIF data
            if e is not None:
                exif = dict(e.items())
                orientation = exif[orientation]

                if orientation == 3:
                    image = image.transpose(PIL.Image.ROTATE_180)
                elif orientation == 6:
                    image = image.transpose(PIL.Image.ROTATE_270)
                elif orientation == 8:
                    image = image.transpose(PIL.Image.ROTATE_90)
    except:
        pass


    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(image)

    number_of_faces = len(face_landmarks_list)
    print("I found {} face(s) in this photograph.".format(number_of_faces))

    # Load the image into a Python Image Library object so that we can draw on top of it and display it
    pil_image = PIL.Image.fromarray(image)

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

    pil_image.save(IMAGE_DIR)

