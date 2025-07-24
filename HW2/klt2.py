
import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt

def denoise_bilateral(img):
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

planes = []
color_plane = []
recency = []

intrinsics = {
    "fx": 748.2795,
    "fy": 748.3063,
    "cx": 490.6477,
    "cy": 506.2725,
    "k1": 0.01912,
    "k2": -0.00521,
    "k3": 0.01841,
    "k4": -0.01079
}
camera_matrix = np.array([
    [intrinsics["fx"], 0, intrinsics["cx"]],
    [0, intrinsics["fy"], intrinsics["cy"]],
    [0, 0, 1]
])
dist_coeffs = np.array([
    intrinsics["k1"],
    intrinsics["k2"],
    intrinsics["k3"],
    intrinsics["k4"]
])

def apply_clahe(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced

def process_image_sequence(input_folder, output_folder, max_corners=500, quality_level=0.01, min_distance=7, ransac_thresh=3.0):

    os.makedirs(output_folder, exist_ok=True)
    
    image_paths = sorted(glob(os.path.join(input_folder, '*.png'))) + \
                 sorted(glob(os.path.join(input_folder, '*.jpg'))) + \
                 sorted(glob(os.path.join(input_folder, '*.jpeg')))
    
    
    lk_params = dict(winSize=(21, 21),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    
    img1 = cv2.imread(image_paths[0])
    h, w = img1.shape[:2]
    new_camera_mtx, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    img1 = cv2.undistort(img1, camera_matrix, dist_coeffs, None, new_camera_mtx)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=5000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # کدک برای فرمت MP4
    fps = 10
    video = cv2.VideoWriter("KLTTracking.mp4", fourcc, fps, (1024, 1024))
     
    for i in tqdm(range(len(image_paths)-1)):
        p0 = cv2.goodFeaturesToTrack(gray1, 
                                    maxCorners=max_corners,
                                    qualityLevel=quality_level,
                                    minDistance=min_distance)

             
        img2 = cv2.imread(image_paths[i+1])
        
        
        img2 = cv2.undistort(img2, camera_matrix, dist_coeffs, None, new_camera_mtx)
        
        if img1 is None or img2 is None:
            continue
        
        
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        

        
        if p0 is None or len(p0) < 10:
            continue
            
       
        p1, st, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)
        
        
        good_old = p0[st == 1]
        good_new = p1[st == 1]
        
        if len(good_old) < 4:
            continue
       
        plane_masks = []
        remaining_pts = good_new.copy()
        used_mask = np.zeros(len(good_new), dtype=bool)
        
        while True:
            if len(remaining_pts) < 4:
                break
                
            H_plane, plane_mask = cv2.findHomography(good_old[~used_mask], remaining_pts, 
                                                    cv2.RANSAC, 0.5)
            if np.sum(plane_mask) < 20:  
                break
                
            full_plane_mask = np.zeros(len(good_new), dtype=bool)
            full_plane_mask[~used_mask] = plane_mask.ravel().astype(bool)
            
            plane_masks.append(full_plane_mask)
            used_mask = used_mask | full_plane_mask
            remaining_pts = good_new[~used_mask]
        
   
        result_img = img2.copy()
  
        gray1 = gray2.copy()
        
    
        for plane_idx, plane_mask in enumerate(plane_masks):
            
        
            flag = 0
            max = 10
            
            keypoints = [cv2.KeyPoint(float(x[1]), float(x[0]), 3) for x in good_new[plane_mask]]
            _, des1 = orb.compute(gray2, keypoints)
            
            for k in range(len(planes)):
                if (recency[k] <= i - 30):
                    continue
                keypoints = [cv2.KeyPoint(float(x[1]), float(x[0]), 3) for x in planes[k]]
                _, des2 = orb.compute(gray2, keypoints)
                
                
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                good_matches = [m for m in matches if m.distance < 32]
                #print("good_matches ",len(planes[k])," match ",len(good_matches))
                
                if len(good_matches)>=max:
                    max = len(good_matches)
                    idx = k
                    flag =1
            max = np.minimum(i+10,20)
            if flag == 1:
                #print("max ",max," idx ",idx)
                color = color_plane[idx]
                planes[idx] = np.concatenate((good_new[plane_mask],planes[idx]))
                recency[idx] = i
            elif flag == 0:
                planes.append(good_new[plane_mask])
                idx= len(planes )-1
                recency.append(i)
                color = tuple(np.random.randint(0, 255, size=3).tolist())
                if color not in color_plane:
                    color_plane.append(color)
                else:
                    color = tuple(np.random.randint(0, 255, size=3).tolist())
                    color_plane.append(color)
                
            
            
            
            for pt in good_new[plane_mask]:
                x, y = pt.ravel()
                cv2.circle(result_img, (int(x), int(y)), 5, color, -1)
                cv2.putText(result_img, f"plane "+ str(idx), 
                   (10, 50*plane_idx+50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
            
            plt.imshow(result_img)
            video.write(result_img)
        plt.axis('off')
        plt.show()
        
        
        output_path = os.path.join(output_folder, f'result_{i:04d}.jpg')
        #cv2.imwrite(output_path, result_img)
    video.release()

input_folder = 'tum_office'  
output_folder = 'output_results'

process_image_sequence(input_folder, output_folder,
                      max_corners=3000,
                      quality_level=0.1,
                      min_distance=3,
                      ransac_thresh=1)
