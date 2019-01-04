import cv2
import os, os.path
import cv2

videos_path = os.path.join(os.path.dirname(__file__), 'VIDEOS_720P_1')

def split_video_into_frames():
	try:
		if not os.path.exists('images'):
			os.makedirs('images')
	except OSError:
		print ('Error: Creating directory of images')
	
	for folder in os.listdir(videos_path):
		folder_path = os.path.join(os.path.dirname(__file__), 'VIDEOS_720P_1/'+folder)
		currentframe = 0
		for video in os.listdir(folder_path):
			print(folder_path)
			print(video)
			cap = cv2.VideoCapture(folder_path+'/'+video)			
			for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
				ret, frame = cap.read()
				if i%20==0:
					name = './images/'+folder+'/frame' + str(currentframe) + '.jpg'
					print ('Creating...' + name)
					cv2.imwrite(name, frame)
				currentframe += 1
			cap.release()
			cv2.destroyAllWindows()	
		
if __name__ == '__main__':
	split_video_into_frames()