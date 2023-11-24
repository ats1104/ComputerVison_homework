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
using std::vector;


using namespace cv;

#ifdef _DEBUG
// debug用のライブラリをリンク
#pragma comment(lib, "opencv_world440d.lib") 
#else
#endif


void test_siftmatch(Mat & img1, Mat& img2) {

	auto detector = SiftFeatureDetector::create(100);

	// detect SIFT keypoints
	vector<KeyPoint> keypoints1, keypoints2;
	detector->detect(img1, keypoints1);
	detector->detect(img2, keypoints2);

	Mat rgb;
	drawKeypoints(img1, keypoints1, rgb, CV_RGB(255, 0, 0));
	imshow("keypoint1", rgb);
	imwrite("keypoint1.png", rgb);
	
	drawKeypoints(img2, keypoints2, rgb, CV_RGB(255, 0, 0));
	imshow("keypoint2", rgb);
	imwrite("keypoint2.png", rgb);

	// compute SIFT descriptor
	auto descriptor = SiftDescriptorExtractor::create();
	Mat descriptor1, descriptor2;
	descriptor->compute(img1, keypoints1, descriptor1);
	descriptor->compute(img2, keypoints2, descriptor2);

	// matching descriptor
	BFMatcher matcher;
	vector<DMatch> matches;
	matcher.match(descriptor1, descriptor2, matches);

	// remove wrong matches
	vector<DMatch> good_matches;
	for (int i = 0; i < matches.size(); i++){
		if (matches[i].distance < 100.0){
			good_matches.push_back(matches[i]);
		}
	}
	// draw result
	drawMatches(img1, keypoints1, img2, keypoints2, good_matches, rgb);
	imshow("match", rgb);
	imwrite("match.png", rgb);
    // 対応点系列を作成 good_matchから対応する点の画像１と画像２のインデックスをとってこれる
    vector<Point2f> points1, points2;
    for (int i = 0; i < good_matches.size(); i++) {
        points1.push_back(keypoints1[good_matches[i].queryIdx].pt);
        points2.push_back(keypoints2[good_matches[i].trainIdx].pt);
    }
    // 変換行列
    Mat A = findHomography(points1, points2, RANSAC);
	// Mat result_img = Mat(1500, 2000, CV_8U);
	Mat result_img = Mat::zeros(Size(1500, 1500), img2.type());
	// result_img = 0;
    Mat warp_img;
    warpPerspective(img1, warp_img, A, Size(1500, 1500));
	imshow("img1toimg2", warp_img);
	imwrite("img1toimg2.png", warp_img);
    

    //射影変換して．合成する．

	for (int y = 0; y < img2.rows; y++) {
		for (int x = 0; x < img2.cols; x++) {
			// warp_img.at<unsigned char>(y, x) = img2.at<unsigned char>(y, x);
			warp_img.at<cv::Vec3b>(y, x) = img2.at<cv::Vec3b>(y, x);

		}
	}
	// for (int y=0; y < warp_img.rows; y++){
	// 	for (int x=0; x < warp_img.cols; x++){
	// 		result_img.at<cv::Vec3b>(y, x) = warp_img.at<cv::Vec3b>(y, x);
	// 	}
	// }

	
	imshow("result", warp_img);
	imwrite("result.png", warp_img);


    //境界とかで平均取るのがええんかね．

    
}

void resize_img(Mat & img, Mat & resize_img){

    int w = img.cols, h = img.rows;
    int sw = 800, sh = 800 * h/w;
    resize(img, resize_img, Size(sw, sh));
}

int main() {
    
	Mat img1, img2, out1, out2;
    // input img
    img1 = imread("input1.jpg");
    resize_img(img1, img1);

    img2 = imread("input2.jpg");
    resize_img(img2, img2);


	// gen img2 from img by scaled & rotating
	// Mat img2;


	test_siftmatch(img1, img2);  
    waitKey(0);
    }
    


