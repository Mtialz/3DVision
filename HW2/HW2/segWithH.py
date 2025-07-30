import cv2
import numpy as np
import os
from glob import glob
from collections import defaultdict
from itertools import combinations
from matplotlib import pyplot as plt


lk_params = dict(winSize=(21, 21),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

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

def mat_dist(matrices,mask1_stack, distance_metric='frobenius'):
    print(len(matrices),len(mask1_stack))
    unique_matrices = matrices.copy()
    unique_mask = mask1_stack.copy()
    i = 0
    #print("before ",mask1_stack)

    while i < len(unique_matrices):
        j = i + 1
        while j < len(unique_matrices):
            
            if distance_metric == 'frobenius':
                dist = np.linalg.norm(unique_matrices[i] - unique_matrices[j], 'fro')
            elif distance_metric == 'spectral':
                dist = np.linalg.norm(unique_matrices[i] - unique_matrices[j], 2)
            elif distance_metric == 'manhattan':
                dist = np.sum(np.abs(unique_matrices[i] - unique_matrices[j]))
            else:
                raise ValueError("متریک فاصله نامعتبر است. گزینه‌های مجاز: 'frobenius', 'spectral', 'manhattan'")

            
            print("dist ",i , j, dist)
            if dist < 20:
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(unique_mask[i])
                axes[0].set_title('تصویر اول')
                axes[0].axis('off') 
                unique_mask[i] = (unique_mask[i] | unique_mask[j]).astype(np.uint8)
                
                 
                axes[1].imshow(unique_mask[i])
                axes[1].set_title('تصویر دوم')
                axes[1].axis('off')  
                plt.tight_layout()  
                plt.show()
            
                unique_matrices[i] = (unique_matrices[j] + unique_matrices[i])/2
                del unique_matrices[j]
                del unique_mask[j]
                print("remove ",i,"j ",j,"matrix ",dist)
            else:
            
                j += 1
        i += 1
    
    return unique_matrices,unique_mask

def detect_planes_from_homographies(homographies, gray1, gray2,copy2, min_inliers=10, ransac_thresh=5.0):
    '''
    # تشخیص ویژگی‌ها با SIFT
    sift = cv2.SIFT_create(1000)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # تطابق ویژگی‌ها
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # فیلتر تطابق‌های خوب
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # تبدیل نقاط به آرایه NumPy
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    '''
    p0 = cv2.goodFeaturesToTrack(gray1,
                                    maxCorners=2000,
                                    qualityLevel=0.1,
                                    minDistance=5)

        
    p1, st, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

        
    pts1 = p0[st == 1].reshape(-1, 1, 2)
    pts2 = p1[st == 1].reshape(-1, 1, 2)

    
    plane_masks = []

    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]

    for i, H in enumerate(homographies):
        
        transformed_pts = cv2.perspectiveTransform(pts1, H)

        
        distances = np.linalg.norm(transformed_pts - pts2, axis=2)

        
        mask = distances < ransac_thresh
        inliers = np.sum(mask)

        if inliers >= min_inliers:
            plane_masks.append(mask.ravel())

           
            color = tuple(np.random.randint(0, 255, 3).tolist())
            for pt in pts2[mask.ravel()]:
                x, y = pt.ravel()
                cv2.circle(copy2, (int(x), int(y)), 5, color, -1)


    return plane_masks, copy2



def strong_segmentation_gray(gray_img):
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
    gray_enhanced = clahe.apply(gray_img)
    blurred = cv2.GaussianBlur(gray_enhanced, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    num_labels, labels_im = cv2.connectedComponents(cleaned)
    return labels_im

def extract_keypoints_and_matches(img1, img2, mask1=None, mask2=None):
    orb = cv2.ORB_create(500)
    #cv2.imwrite("dfd.jpg",mask1)

    #cv2_imshow( mask1)

    kp1, des1 = orb.detectAndCompute(img1, mask1)
    kp2, des2 = orb.detectAndCompute(img2, mask2)
    if des1 is None or des2 is None:
        return None, None, []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good_matches = [m for m in matches if m.distance < 32]
    matches = sorted(good_matches, key=lambda x: x.distance)
    return kp1, kp2, matches

def extract_keypoints(img1,mask1=None):
    orb = cv2.ORB_create(500)
    #cv2.imwrite("dfd.jpg",mask1)

    #cv2_imshow( mask1)

    kp1, des1 = orb.detectAndCompute(img1, mask1)
    if des1 is None :
        return None, None
    
    return kp1, des1

def compute_homography(kp1, kp2, matches):
    if len(matches) < 4:
        return None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return H, mask

def draw_segment_matches(labels1, matched_segments, output_path):
    out_img = np.zeros((labels1.shape[0], labels1.shape[1], 3), dtype=np.uint8)
    for i, (label1, label2) in enumerate(matched_segments.items()):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        out_img[labels1 == label1] = color
    cv2.imwrite(output_path, out_img)

folder_path = 'tum_office'
image_paths = sorted(glob(os.path.join(folder_path, '*.jpg')) )


img1 = cv2.imread(image_paths[0])
copy1 = img1.copy()
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    
labels1 = strong_segmentation_gray(img1)
label1_ids = set(np.unique(labels1)) - {0}
h, w = img1.shape[:2]
new_camera_mtx, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
img1 = cv2.undistort(img1, camera_matrix, dist_coeffs, None, new_camera_mtx)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # کدک برای فرمت MP4
fps = 10
video = cv2.VideoWriter("maskWOH.mp4", fourcc, fps, (w, h))

for i in range(1,len(image_paths) - 1):
    print(f"\n🔍 Comparing Frame {i} with Frame {i+1}")

    img2 = cv2.imread(image_paths[i])
    img2 = cv2.undistort(img2, camera_matrix, dist_coeffs, None, new_camera_mtx)
    copy2 = img2.copy()
    
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


    labels2 = strong_segmentation_gray(img2)

    matched_segments = dict()

    label2_ids = set(np.unique(labels2)) - {0}

    best_Homo = []
    mask1_stack = []
    for label1 in label1_ids:
        mask1Orig = None
        mask_best1 = None
        mask_best2 = None
        mask1 = (labels1 == label1).astype(np.uint8) * 255

        if (np.sum(mask1) / 255) < 4000:
            continue


        max_mask = 0
        for label2 in label2_ids:

            mask2 = (labels2 == label2).astype(np.uint8) * 255
            mask3= mask2 & mask1

            if (np.sum(mask3) > max_mask):
                max_mask =  np.sum(mask3)
                mask_best2 = mask2.copy()
                mask_best1 = mask1.copy()
                mask1Orig = mask1.copy()
                
            

        if   max_mask > 100:
            '''
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(mask_best2)
            axes[0].set_title('تصویر اول')
            axes[0].axis('off')  # غیرفعال کردن محورها
            axes[1].imshow(mask_best1)
            axes[1].set_title('تصویر دوم')
            axes[1].axis('off')  # غیرفعال کردن محورها
            plt.tight_layout()  # تنظیم خودکار فاصله‌ها برای جلوگیری از همپوشانی
            plt.show()
            '''
            rows = np.any(mask_best1 == 255, axis=1)
            cols = np.any(mask_best1 == 255, axis=0)


            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            mask_best1[rmin:rmax+1, cmin:cmax+1] = 255

            rows = np.any(mask_best2 == 255, axis=1)
            cols = np.any(mask_best2 == 255, axis=0)

            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            mask_best2[rmin:rmax+1, cmin:cmax+1] = 255

            kp1, kp2, matches = extract_keypoints_and_matches(img1, img2, mask_best1, mask_best2)
            if matches and len(matches) > 40:

                H, mask_inliers = compute_homography(kp1, kp2, matches)

                if H is not None and np.sum(mask_inliers) >= 30:
                    best_Homo.append(H)
                    mask1_stack.append(mask1Orig)

    homographies,masks = mat_dist(best_Homo,mask1_stack, 'frobenius')

    #print("homo ",len(homographies),len(masks))
    for k in range(len(masks)):
        '''
        plt.figure(figsize=(12, 6))
        plt.imshow(masks[i])
        plt.axis('off') 
                 # غیرفعال کردن محورها
        plt.tight_layout()  # تنظیم خودکار فاصله‌ها برای جلوگیری از همپوشانی
        plt.show()
        '''
        kp1,decs = extract_keypoints(copy1,masks[k])
        rows = np.any(masks[k] == 255, axis=1)
        cols = np.any(masks[k] == 255, axis=0)


        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        print(" rmin, rmax ",rmin, rmax,cmin, cmax)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        copy1 = cv2.drawKeypoints(
            copy1, 
            kp1, 
            None, 
            color=color,  
            flags=0)
        cv2.rectangle(copy1, (cmin, rmin), (cmax, rmax),color, 5)
            
   
    output_path = f"matched_segments_frame{i}.png"
    #draw_segment_matches(labels1, matched_segments, output_path)
    print(f"📁 Saved matched segments visualization to {output_path}")
    #plane_masks, result_img = detect_planes_from_homographies(homographies, img1, img2,copy2)

    
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(copy1, cv2.COLOR_BGR2RGB))
    video.write(cv2.cvtColor(copy1, cv2.COLOR_BGR2RGB))
    #plt.title(f"Detected {len(plane_masks)} Planes")
    plt.axis('off')
    plt.show()

    
    output_path = os.path.join('outN', f'{i:04d}.png')
    cv2.imwrite(output_path, copy1)
    img1= img2
    labels1 = labels2
    label1_ids = label2_ids
    copy1 = copy2
video.release()
