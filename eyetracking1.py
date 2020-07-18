# USAGE
# python eyetracking.py --face cascades/haarcascade_frontalface_default.xml --eye cascades/haarcascade_eye.xml --video video/adrian_eyes.mov
# python eyetracking.py --face cascades/haarcascade_frontalface_default.xml --eye cascades/haarcascade_eye.xml

# import the necessary packages
from pyimagesearch.eyetracker import EyeTracker
from pyimagesearch import imutils
import argparse
import cv2
import numpy as np
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required = True,
	help = "path to where the face cascade resides")
ap.add_argument("-e", "--eye", required = True,
	help = "path to where the eye cascade resides")
ap.add_argument("-v", "--video",
	help = "path to the (optional) video file")
args = vars(ap.parse_args())

# construct the eye tracker
et = EyeTracker(args["face"], args["eye"])
imeye=cv2.imread("eye.jpg")

# if a video path was not supplied, grab the reference
# to the gray
if not args.get("video", False):
	camera = cv2.VideoCapture(0)

# otherwise, load the video
else:
	camera = cv2.VideoCapture(args["video"])

# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()

	# if we are viewing a video and we did not grab a
	# frame, then we have reached the end of the video
	if args.get("video") and not grabbed:
		break

	# resize the frame and convert it to grayscale
	frame = imutils.resize(frame, width = 300)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces and eyes in the image
	rects = et.track(gray)

	# loop over the face bounding boxes and draw them
	for (i,rect) in enumerate (rects):
		if i==0:
			fW=rect[2]-rect[0]
			fH=rect[3]-rect[1]
			x=(fW//2)+rect[0]
			y=(fH//2)+rect[1]
			r=y-rect[1]
			blur=cv2.medianBlur(frame,9)
			cv2.circle(frame,(x,y),r,(0,0,255),2)
			mask=np.zeros((frame.shape[:2]),dtype="uint8")
			cv2.circle(mask,(x,y),r,(255,255,255),-1)
			blur[np.where(mask == 255)] = frame[np.where(mask == 255)]
		if i==1:
			fW=rect[2]-rect[0]
			fH=rect[3]-rect[1]
			x=(fW//2)+rect[0]
			y=(fH//2)+rect[1]
			eye1=frame[rect[1]:rect[3],rect[0]:rect[2]]
			imeyeresize=imutils.resize(imeye, width = fW)
			x_offset=rect[0]
			y_offset=rect[1]
			blur[y_offset:y_offset+imeyeresize.shape[0], x_offset:x_offset+imeyeresize.shape[1]] = imeyeresize
			eye1=cv2.resize(eye1,(frame.shape[1],frame.shape[0]))
			eye12=np.hstack([blur,eye1])
			cv2.imshow("eye", eye12)
		if i==2:
			fW=rect[2]-rect[0]
			fH=rect[3]-rect[1]
			x=(fW//2)+rect[0]
			y=(fH//2)+rect[1]
			eye2=frame[rect[1]:rect[3],rect[0]:rect[2]]
			eye2=cv2.resize(eye2,(frame.shape[1],frame.shape[0]))
			x_offset=rect[0]
			y_offset=rect[1]
			blur[y_offset:y_offset+imeyeresize.shape[0], x_offset:x_offset+imeyeresize.shape[1]] = imeyeresize
			eye2=np.hstack([blur,eye1,eye2])
			cv2.imshow("eye", eye2)

	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()