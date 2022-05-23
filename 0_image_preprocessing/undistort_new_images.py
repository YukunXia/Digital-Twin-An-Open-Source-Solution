import cv2
import numpy as np
import os
import glob
import copy
from tqdm import tqdm

mtx_vertical = np.load('mtx.npy')
mtx_horizontal = np.copy(mtx_vertical)
mtx_horizontal[0,0] = mtx_vertical[1,1]
mtx_horizontal[0,-1] = mtx_vertical[1,-1]
mtx_horizontal[1,1] = mtx_vertical[0,0]
mtx_horizontal[1,-1] = mtx_vertical[0,-1]

assert(mtx_vertical[0,1] == 0.0)

dist = np.load('dist.npy')

rootdir = './data_orig'
w_h_ratio = 4/3
w_new = 960
h_new = 720

newcameramtx = None
roi = None

for file in os.listdir(rootdir):

    sub_folder = os.path.join(rootdir, file)

    is_horizontal = None
    sub_folder_exists = None
    new_cam_mtx_inited = False
    if os.path.isdir(sub_folder):
        if sub_folder_exists == None:
            new_sub_folder = copy.copy(sub_folder)
            new_sub_folder = new_sub_folder.split("/")
            new_sub_folder[1] = "data"
            new_sub_folder = "/".join(new_sub_folder)
            print("new_sub_folder = ", new_sub_folder)
            if not os.path.isdir(new_sub_folder):
                os.mkdir(new_sub_folder)
            sub_folder_exists = True

        image_fnames = glob.glob(sub_folder + "/*.jpg")
        image_fnames.sort()

        assert(len(image_fnames) < 10000)
        for img_index, image_fname in tqdm(enumerate(image_fnames)):
            img = cv2.imread(image_fname)
            h,w = img.shape[:2]

            img_fname_split = image_fnames[0].split("/")
            img_fname_split[1] = "data"

            assert(w/h == 4/3 or h/w == 4/3)
            if is_horizontal == None:
                is_horizontal = (w/h == 4/3)
                print("is_horizontal = ", is_horizontal)
            elif is_horizontal == False:
                assert(h/w == 4/3)
            else:
                assert(w/h == 4/3)

            if not new_cam_mtx_inited:
                # alpha = 1 -> the image is resized so all original pixels fit into the image plane, thus you will most likely get invalid pixels in your undistorted image
                # alpha = 0 -> the image is resized so there are no invalid pixels in the image plane, thus you will most likely lose valid pixels in your undistorted image
                alpha = 0

                assert(w_new / w == h_new / h)
                factor = w_new / w

                if is_horizontal:
                    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx_horizontal, dist, (w,h), alpha, (w, h))
                    mtx = np.copy(mtx_horizontal)
                else:
                    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx_vertical, dist, (w,h), alpha, (w, h))
                    mtx = np.copy(mtx_vertical)

                newcameramtx_resize = np.copy(newcameramtx)
                newcameramtx_resize[:2] *= factor

                new_camera_mtx_fname = img_fname_split.copy()
                new_camera_mtx_fname[-1] = "camera.txt"
                new_camera_mtx_fname = "/".join(new_camera_mtx_fname)
                np.savetxt(new_camera_mtx_fname, newcameramtx_resize)

                new_cam_mtx_inited = True

            img_undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)

            img_undistorted = cv2.resize(img_undistorted, (w_new, h_new))

            img_fname_split[-1] = str(img_index).zfill(4)+".jpg"
            new_img_fname = "/".join(img_fname_split)
            cv2.imwrite(new_img_fname, img_undistorted)