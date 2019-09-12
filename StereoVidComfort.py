import ffmpeg
import numpy as np
import matplotlib
import cv2
import os
import sys
from matplotlib import pyplot as plt
from scipy import stats


'''
TODO:
    0:读取视频  √
    1:获取视差  √
    2:获取运动矢量    √
    3:确定舒适度     √
    4:加舒适度水印    （不做)
    5:提高舒适度     √（估计可提高的值）
    ...
'''

# 打开视频文件
def openVid():
    fileName = input("video path: ./vid/")
    fileName = "./vid/" + fileName
    while not os.path.isfile(fileName):
        if os.path.isfile(fileName + ".mkv"):
            fileName = fileName + ".mkv"
            break
        print("file doesn't exist!")
        fileName = input("video path: ./vid/")
        fileName = "./vid/" + fileName
    cap = cv2.VideoCapture(fileName)
    if cap.isOpened():
        return cap
    else:
        print("cannot open video.")
        sys.exit()


# 获取视频总帧数
def getFrameCount(cap):
    if cap.isOpened():
        return cap.get(cv2.CAP_PROP_FRAME_COUNT)
    else:
        print("cannot open video.")
        sys.exit()

# 获取帧速率
def getFrameRate(cap):
    if cap.isOpened():
        return cap.get(cv2.CAP_PROP_FPS)
    else:
        print("cannot open video.")
        sys.exit()

# 给出左右画面，计算景深
def getDepthMap(imgL, imgR):

    # stereo = cv2.StereoBM_create(numDisparity = 32, blockSize = 3)        # 速度快，准确性较低，单通道
    stereo = cv2.StereoSGBM_create(
        minDisparity=-16, numDisparities=48, blockSize=5, P1=320, P2=1280)      # 速度稍慢，准确性较高，多通道
    return stereo.compute(imgL, imgR)


# 给出前后两帧，计算帧间运动矢量
def getMotionVector(prvs, next):
    hsv = np.zeros_like(imgR)  # 将运动矢量按hsv显示，以色调h表示运动方向，以明度v表示运动位移
    hsv[..., 1] = 255  # 饱和度置为最高

    # 转为灰度以计算光流
    prvs = cv2.cvtColor(prvs, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)  # 计算两帧间的光流，即运动矢量的直角坐标表示
    mag, ang = cv2.cartToPolar(
        flow[..., 0], flow[..., 1])  # 运动矢量的直角坐标表示转换为极坐标表示
    hsv[..., 0] = ang*180/np.pi/2  # 角度对应色调
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # 位移量对应明度
    return hsv


if __name__ == "__main__":
    cap = openVid()
    isDemo = int(input("is Demo(0/1)?"))
    calcMod = int(input("calc optimize potential?"))
    frameRate = getFrameRate(cap)
    frameCount = getFrameCount(cap)
    framesCalculated = 0
    framesOptimized = 0
    framesComfort = []
    framesComfortOptimized = []

    isSuccess, img = cap.read()
    if not isSuccess:
        print("video read error.")
        sys.exit()

    # 分割左右画面
    imgL = np.split(img, 2, 1)[0]
    imgR = np.split(img, 2, 1)[1]
    prvs = imgR  # 上一帧的右画面，用于运动矢量计算

    # 每秒取5帧进行计算
    for frameID in range(round(0), round(frameCount), round(frameRate/5)):
        if frameID >= frameCount - 3:
            frameID = frameCount - 3
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameID)
        isSuccess, img = cap.read()
        if not isSuccess:
            print("video read error.")
            break

        # 分割左右画面
        imgL = np.split(img, 2, 1)[0]
        imgR = np.split(img, 2, 1)[1]

        next = imgR  # 当前帧的右画面，用于运动矢量计算
        hsv = getMotionVector(prvs, next)
        hsv_bak = hsv.copy()
        # 计算深度图,disparity越大，景深越小，物体越近
        disparity = getDepthMap(imgL, imgR)

        framesCalculated += 1
        comfort = 1

        # 显示计算结果
        print("time: ", round(frameID/frameRate, 2))

        # 景深的平均值，偏大则意味着负视差（出屏感），可能不适
        AVG_depth = round(np.mean(disparity), 2)
        print("AVG depth: ", AVG_depth)      # 大于-10时开始不适，权重为0.15
        if AVG_depth > -10:
            comfort -= 0.15

        # 运动矢量大小的平均值，可判断画面大致上是否稳定
        AVG_motionMag = round(np.mean(hsv[..., 2]), 2)
        print("AVG motionMag: ", AVG_motionMag)       # 大于20时略不适，权重0.1
        if AVG_motionMag > 20:
            comfort -= 0.1

        # 景深的众数，由于景深基本不连续，众数意义不大
        # print("Mode depth: ", stats.mode(disparity.reshape(-1))[0][0])      # 无明显阈值

        # 运动矢量大小的众数，一般为0，若较大，说明画面中存在较大面积的快速运动，可能不适
        Mode_motionMag = stats.mode(hsv[..., 2].reshape(-1))[0][0]
        # 大于0则不适，越大越不适，权重0.2，0到30归一化为0.1到0.15，大于30为0.2
        print("Mode motionMag: ", Mode_motionMag)
        if Mode_motionMag > 0:
            if Mode_motionMag > 30:
                comfort -= 0.2
            else:
                comfort -= (Mode_motionMag/600 + 0.1)

        # 景深的标准差，若偏大说明景深范围较大，可能不适，但同时也是3D感更强的特征
        STD_depth = round(np.std(disparity), 2)
        print("STD depth: ", STD_depth)        # 大于130时略不适，权重为0.15
        if STD_depth > 130:
            comfort -= 0.15

        # 运动矢量大小的标准差，若偏大说明各部分运动比较不一致，可能需要结合运动矢量的方向作进一步判断，若存在较复杂的运动形式，则可能不适
        STD_motionMag = round(np.std(hsv[..., 2]), 2)
        print("STD motionMag: ", STD_motionMag)       # 大于20时略不适，权重为0.1
        if STD_motionMag > 20:
            comfort -= 0.1

        # 运动矢量方向的标准差，若偏大说明各部分运动比较不一致，可能需要结合运动矢量的大小作进一步判断，若存在较复杂的运动形式，则可能不适
        # print("STD motionAng: ", round(np.std(hsv[...,0]),2))       # 无明显阈值

        disparity_Positive = disparity.copy()
        disparity_Positive[disparity_Positive < 0] = 0

        # 负视差的像素的所占比例，大于0.2时比较不适，权重0.15，0.2到0.4归一化为0.05到0.1，大于0.4为0.15
        PCT_disparity_Positive = np.count_nonzero(
            disparity_Positive)/disparity_Positive.shape[0]/disparity_Positive.shape[1]
        print("close pixels percetage:", round(PCT_disparity_Positive, 3))
        if PCT_disparity_Positive > 0.2:
            if PCT_disparity_Positive > 0.4:
                comfort -= 0.15
                orgn_cmft = -0.15
            else:
                comfort -= ((PCT_disparity_Positive - 0.2) / 4 + 0.05)
                orgn_cmft = -((PCT_disparity_Positive - 0.2) / 4 + 0.05)
            if calcMod:
                # 视差重映射并重新计算
                # 实际并不写入文件，只估计此项提升值
                trans = np.float32([[1,0,20],[0,1,0]])
                imgR_Mod = cv2.warpAffine(imgR, trans, imgR.shape[:2])
                imgR_Mod = imgR_Mod.transpose((1,0,2))
                disparity_Mod = getDepthMap(imgL, imgR_Mod)
                disparity_Positive = disparity_Mod.copy()
                disparity_Positive[disparity_Positive < 0] = 0
                PCT_disparity_Positive = np.count_nonzero(
                    disparity_Positive)/disparity_Positive.shape[0]/disparity_Positive.shape[1]
                print("Modified close pixels percetage:", round(PCT_disparity_Positive, 3))
                if PCT_disparity_Positive > 0.2:
                    if PCT_disparity_Positive > 0.4:
                        mod_cmft = -0.15
                    else:
                        mod_cmft = -((PCT_disparity_Positive - 0.2) / 4 + 0.05)
                else:
                    mod_cmft = 0
                comfort_optimized = round(mod_cmft - orgn_cmft, 3)
                if comfort_optimized > 0:
                    framesOptimized += 1
                    framesComfortOptimized.append(comfort_optimized)
                    print("comfort can be optimized by ", comfort_optimized)



        # 存在运动的像素点的视差平均值
        movingPixels = hsv[..., 2]
        movingPixels[movingPixels < 10] = 0     # 小于10的运动认为是静止
        movingPixels[movingPixels > 0] = 1
        movingDepth = np.multiply(disparity, movingPixels)
        AVG_movingDepth = round(np.sum(movingDepth) /
                                np.count_nonzero(movingDepth))
        print("AVG movingDepth: ", AVG_movingDepth)        # 大于5时不适，权重0.15
        if AVG_movingDepth > 5:
            comfort -= 0.15

        framesComfort.append(comfort)
        comfort = round(comfort, 3)

        print()
        print("CurFrameComfort: ", comfort)
        print("TotalComfort: ", round(sum(framesComfort)/framesCalculated, 2))
        print()

        # 当为demo模式时显示当前帧画面、运动矢量图和景深图
        if isDemo:
            # 显示当前帧
            cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            cv2.imshow('img', img)
            cv2.waitKey(1)

            # cv2.namedWindow("imgL", cv2.WINDOW_NORMAL)
            # cv2.imshow('imgL', imgL)
            # cv2.namedWindow("imgR", cv2.WINDOW_NORMAL)
            # cv2.imshow('imgR', imgR)

            # 显示当前帧的运动矢量的hsv表示
            bgr = cv2.cvtColor(hsv_bak, cv2.COLOR_HSV2BGR)  # hsv转为rgb用于显示
            cv2.namedWindow("MotionVector", cv2.WINDOW_NORMAL)
            cv2.imshow("MotionVector", bgr)
            cv2.waitKey(1)
            # 显示当前帧的景深图
            plt.title("DepthMap")
            plt.imshow(disparity)
            #  plt.pause(0.1)
            #  input("press Enter to continue")

            # 运动矢量的直方图，方便查看数值
            # plt.title("MotionVector")
            # plt.imshow(hsv[...,2])
            # plt.show()
            plt.pause(0.1)
            input("press Enter to continue")
        prvs = next  # 当前帧覆盖上一帧，继续计算
    print("TotalFrameCalculated: ", framesCalculated)
    print("TotalComfort: ", round(sum(framesComfort)/framesCalculated, 2))
    if calcMod:
        print("estimated comfort optimization potential：", round(sum(framesComfortOptimized)/framesOptimized, 3))
    print("success")
