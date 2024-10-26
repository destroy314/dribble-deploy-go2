import cv2
import numpy as np
import glob
import pickle

def get_K_and_D(checkerboard, imgsPath):
    CHECKERBOARD = checkerboard
    square_size = 45 / 1000  # 每个格子的大小为30mm，转换为米
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size

    _img_shape = None
    objpoints = []
    imgpoints = []
    images = glob.glob(imgsPath + '/*.jpg')  # 假定棋盘格图像为 PNG 格式；可以根据实际修改

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"无法读取图像: {fname}")
            continue
            
        if _img_shape is None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret:
            objpoints.append(objp)
            cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
            imgpoints.append(corners)

    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]

    rms, _, _, _, _ = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

    DIM = _img_shape[::-1]
    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM=" + str(DIM))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")
    
    return DIM, K, D

def undistort(img_path, K, D, DIM, scale=0.6, imshow=False):
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图像: {img_path}")
        return None
        
    dim1 = img.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort
    
    # 确保输入图像的纵横比与标定图像一致
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have the same aspect ratio as the ones used in calibration"
    
    if dim1[0] != DIM[0]:
        img = cv2.resize(img, DIM, interpolation=cv2.INTER_AREA)
    
    Knew = K.copy()
    if scale:  # 改变视场
        Knew[(0, 1), (0, 1)] = scale * Knew[(0, 1), (0, 1)]
        
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Knew, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    if imshow:
        cv2.imshow("Undistorted", undistorted_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return undistorted_img

if __name__ == '__main__':
    # 计算 K 和 D，使用棋盘格内角点 (8, 11) 的图像进行标定
    DIM, K, D = get_K_and_D((5, 8), '/home/hgd/data/calibration_1009')
    #保存畸变参数
    with open('/home/hgd/data/calibration_1009/calibration_params.pickle', 'wb') as f:
        pickle.dump({'K': K, 'D': D,'DIM':DIM}, f)
    print("内参和畸变参数保存成功。")

    # 读取需要去畸变的测试图像
    test_image_path = '/home/hgd/data/pic/001.jpg'  # 替换为你的测试图像路径
    undistorted_img = undistort(test_image_path, K, D, DIM, scale=1.0, imshow=True)
