#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

#ifdef _DEBUG
// debug用のライブラリをリンク
#pragma comment(lib, "opencv_world440d.lib") 
#else
// Release用のライブラリをリンク
#pragma comment(lib, "opencv_world440.lib") 

#endif
void proc(Mat& img, Mat& out){
    
    int w = img.cols, h = img.rows;
    int sw = 800, sh = 800 * h/w;
    resize(img, img, Size(sw, sh));
    int dw = sw/10, dh = sh/10;

    img.copyTo(out);
    for (int j = 0; j < 10; j+=2){
        for (int i = 0; i < 10; i+=2){
            rectangle(out, Point(j*dw, i*dh), Point((j+1)*dw, (i+1)*dh), Scalar(0, 0, 255), -1);
            rectangle(out, Point((j+1)*dw, (i+1)*dh), Point((j+2)*dw, (i+2)*dh), Scalar(0, 0, 255), -1);            
        }

    }
    String text = "231d-8746";
    cv::putText(out, text, cv::Point(25,75), cv::FONT_HERSHEY_SIMPLEX, 2.5, cv::Scalar(255,0,0), 3);

}
int main() {
    
	Mat img;
    img = imread("in.jpg");
    
    Mat out;
    //10*10の市松模様を描く
    proc(img, out);
	imshow("out", out);
	imwrite("out.jpg", out);
	waitKey(0);
}
