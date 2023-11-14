#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <limits>


using std::cout; using std::cin;
using std::endl; using std::string;
using std::to_string;


using namespace cv;

#ifdef _DEBUG
// debug用のライブラリをリンク
#pragma comment(lib, "opencv_world440d.lib") 
#else
// Release用のライブラリをリンク
#pragma comment(lib, "opencv_world440.lib") 

#endif

int main() {
    
	Mat img, gray, canny, r_canny, bilateral, out; 
    img = imread("in.jpg");
    // resize
    int w = img.cols, h = img.rows;
    int sw = 800, sh = 800 * h/w;
    resize(img, img, Size(sw, sh));
    cvtColor(img, gray, COLOR_BGR2GRAY);
    // Cannyのエッジフィルタ．
    Canny(gray, canny, 100, 200);
    // 色反転
    bitwise_not(canny, r_canny);
    // 3チャネルに
    cvtColor(r_canny, r_canny, COLOR_GRAY2BGR);
    bilateralFilter(img, bilateral, -1,  70, 75);
    // ビット単位の論理積：r_cannyの黒（０）部分だけ，0に．
    bitwise_and(bilateral, r_canny, out);
    // 描画・保存
    // imshow("gray", gray);
    // imshow("r_canny", r_canny);
    // imshow("out", out);
    // imshow("bil", bilateral);   
    imwrite("r_canny.jpg", r_canny);
    imwrite("bil.jpg", bilateral);
    imwrite("out.jpg", out);
 
	waitKey(0);
}
