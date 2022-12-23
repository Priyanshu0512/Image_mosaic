import numpy as np
import cv2
import imutils
import os

images_folder = 'Photos'
myfolders = os.listdir(images_folder)
print(myfolders)

# EXTRACTING IMAGES FROM MULTIPLE FOLDERS.
for folder in myfolders:
    path = images_folder + '/' + folder
    images = []
    images_gray = []
    images_list = os.listdir(path)
    print(images_list)
    print(f'Total number of images detected {len(images_list)}')
    for image_name in images_list:
        current_image = cv2.imread(f'{path}/{image_name}')
        current_image = cv2.resize(current_image, (0, 0), None, 0.5, 0.5)
        # current_image_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        images.append(current_image)
        # images_gray.append(current_image_gray)

    stitcher = cv2.Stitcher_create()
    (status, stitched_img) = stitcher.stitch(images)
    if status == cv2.STITCHER_OK:
        stitched_img = cv2.copyMakeBorder(stitched_img, 15, 15, 15, 15, cv2.BORDER_CONSTANT, (0, 0, 0))
        grayscale_img = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
        thresh_img = cv2.threshold(grayscale_img, 0, 255, cv2.THRESH_BINARY)[1]

# GRABBING ALL CONTOURS TO DETERMINE THE LARGEST CONTOUR.
        contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        max_area = max(contours, key=cv2.contourArea)

        mask = np.zeros(thresh_img.shape, dtype="uint8")
        starting_x, starting_y, width, height = cv2.boundingRect(max_area)
        cv2.rectangle(mask, (starting_x, starting_y), (starting_x+width, starting_y+height), 255, -1)
        minRectangle = mask.copy()
        temp_mask = mask.copy()

        while cv2.countNonZero(temp_mask) > 0:
            minRectangle = cv2.erode(minRectangle, None)
            temp_mask = cv2.subtract(minRectangle, thresh_img)

        contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = imutils.grab_contours(contours)
        max_area = max(contours, key=cv2.contourArea)

        starting_x, starting_y, width, height = cv2.boundingRect(max_area)
        stitched_img = stitched_img[starting_y:starting_y+height, starting_x:starting_x+width]
        cv2.imshow("Stitched Image Processed", stitched_img)
        cv2.waitKey(0)
        print("Panorama created")

    else:
        print("Panorama generation unsuccessfully")
