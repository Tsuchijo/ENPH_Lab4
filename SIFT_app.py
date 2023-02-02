#!/usr/bin/env python3
from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import numpy as np  # numpy
import sys

class My_App(QtWidgets.QMainWindow):
    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)

        #create sift object
        self._sift = cv2.SIFT_create()

        # Feature matching
        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        self._flann = cv2.FlannBasedMatcher(index_params, search_params)

        self._cam_id = 0
        self._cam_fps = 2
        self._is_cam_enabled = False
        self._is_template_loaded = False

        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        self._camera_device = cv2.VideoCapture(self._cam_id)
        self._camera_device.set(3, 320)
        self._camera_device.set(4, 240)

        self.browse_button.clicked.connect(self.SLOT_browse_button)

        # Timer used to trigger the camera
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(100 / self._cam_fps)

    
    def SLOT_browse_button(self):  
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if dlg.exec():
            self.template_path = dlg.selectedFiles()[0]
        
        pixmap = QtGui.QPixmap(self.template_path)
        # use sift to detect keypoints and descriptors
        self.template_kpm, self.template_desc, self.template_img = self.detect_features(self.convert_pixmap_to_cv(pixmap))
        self.template_label.setPixmap(self.convert_cv_to_pixmap(self.template_img))
        print("Loaded template image: " + self.template_path)
    
    # Source: stackoverflow.com/questions/34232632/
    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height, 
                        bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)

    # Source: copilot
    def convert_pixmap_to_cv(self, pixmap):
        image = pixmap.toImage()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(image.height(), image.width(), 4)
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)

    def SLOT_query_camera(self):
        ret, frame = self._camera_device.read()

        #convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #detect features
        frame_kp, frame_desc = self._sift.detectAndCompute(gray_frame, None)

        #compare features to uploaded template
        matches = self._flann.knnMatch(self.template_desc, frame_desc, k=2)
        #search for good matches
        good_points = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_points.append(m)
        
        #draw bounding box
        frame = self.compute_homography(self.template_kpm, frame_kp, good_points, frame, self.template_img)
        pixmap = self.convert_cv_to_pixmap(frame)
        self.live_image_label.setPixmap(pixmap)

    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")
    
    ## Computes the homography and does the perspective transform
    # @param kp1 The keypoints of the first image
    # @param kp2 The keypoints of the second image
    # @param matches The matches between the keypoints
    # @param frame The frame to draw the bounding box on
    # @param template The template image
    def compute_homography(self, kp1, kp2, matches, frame, template):
        if(len(matches) < 4):
            return frame
        # Create arrays of points for the keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        # Compute the homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        # Perspective transform
        h = template.shape[1]
        w = template.shape[0]
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)      
        dst = cv2.perspectiveTransform(pts, M)
        #define params for drawing matches
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

        # Draw the bounding box
        img2 = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        img2 = cv2.drawMatches(template,kp1,frame,kp2,matches,None,**draw_params)
        return img2

    ## Detects featues using SIFT and OpenCV
    # draws keypoints on the image and displays it 
    # @param img The image to detect features in
    # @return kp The keypoints detected in the image
    # @return desc The descriptors of the keypoints
    def detect_features(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = self._sift.detectAndCompute(gray, None)
        img = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return kp, desc, img
        
        




if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())
