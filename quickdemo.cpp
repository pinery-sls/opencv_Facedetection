//
// Created by pinery on 2021/6/15.
//

#include "quickopencv.h"
#include <iostream>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace std;

void QuickDemo::face_detection_demo() {
    std::string root_dir="/home/pinery/SoftWares/OpenCV/opencv-3.4.10/samples/dnn/face_detector/";
    dnn::Net net = dnn::readNetFromTensorflow(root_dir+"opencv_face_detector_uint8.pb", root_dir+"opencv_face_detector.pbtxt");

    VideoCapture capture("/home/pinery/code/Deploy/opencv/Megamind.avi");
    Mat frame;
    while (true)
    {
        capture.read(frame);
        if (frame.empty())
        {
            break;
        }

        Mat blob = dnn::blobFromImage(frame, 1.0, Size(300, 300), Scalar(107,177,123), false, false);
        net.setInput(blob);
        Mat probs = net.forward(); // NCHW -> 7
        Mat detecionMat(probs.size[2], probs.size[3], CV_32F, probs.ptr<float>());

        // 解析结果 NCHW
        for (int i = 0; i < detecionMat.rows; i++)
        {
            float confidence = detecionMat.at<float>(i, 2);
            if (confidence > 0.5)
            {
                int x1 = static_cast<int>(detecionMat.at<float>(i, 3) * frame.cols);
                int y1 = static_cast<int>(detecionMat.at<float>(i, 4) * frame.rows);
                int x2 = static_cast<int>(detecionMat.at<float>(i, 5) * frame.cols);
                int y2 = static_cast<int>(detecionMat.at<float>(i, 6) * frame.rows);
                Rect box(x1, y1, x2-x1, y2-y1);
//                Rect box1(x1, y1, 30, 30);
                putText(frame, "face", Point (x1, y1-10), FONT_HERSHEY_COMPLEX, 1, Scalar(0,255,255), 2, 1);
                rectangle(frame, box, Scalar(0,0,255), 2, 8, 0);
            }
        }
        imshow("video", frame);
        int c = waitKey(1);
        if (c==27)
        {
            break;
        }
    }
    destroyAllWindows();
}