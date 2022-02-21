import cv2
import os


"""
Class to make reading from video easier.
get_frame: return the frame at any given index
frame_itr: frame iterator to retrieve frames in a loop. syntax is as follows

a = VideoHandler(path)
f_itr = a.frame_itr()
while True:
    img, fno = next(f_itr)
    if not img: break
"""


class VideoHandler:
    def __init__(self, path):
        assert os.path.exists(path), f'Error, {path} does not exist'
        self.path = path
        self.vdo = cv2.VideoCapture(path)
        self.current_frame = -1


    def get_frame(self, fno):
        self.vdo.set(1, fno)
        _, frame = self.vdo.read()
        self.current_frame = fno
        return frame

    def frame_itr(self):
        while self.vdo.grab():
            self.current_frame += 1
            _, img = self.vdo.retrieve()
            yield img, self.current_frame
        # yield False, False



if __name__ == '__main__':
    """
    tests functionality of class
    """

	# cv2.namedWindow("diff", cv2.WINDOW_NORMAL)
	# cv2.resizeWindow("diff", 800, 600)

	# cv2.namedWindow("med", cv2.WINDOW_NORMAL)
	# cv2.resizeWindow("med", 800, 600)

    p = '/home/george/Downloads/car.mp4'
    feed = VideoHandler(p)

    for i in range(100):
        im, f = next(feed.frame_itr())
        cv2.imshow('view', im)
        cv2.waitKey(5)
        print(f)

    print('============')
    im = feed.get_frame(0)
    cv2.imshow('view', im)
    cv2.waitKey(500)
    print('============')

    for i in range(100):
        im, f = next(feed.frame_itr())
        cv2.imshow('view', im)
        cv2.waitKey(5)
        print(f)

