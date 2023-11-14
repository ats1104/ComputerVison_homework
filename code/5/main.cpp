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
#endif

float Box_Muller() {

	float X, Y;
	X = (rand() % 10000) / 10000.0;
	Y = (rand() % 10000) / 10000.0;

	return sqrt(-2 * log(X)) * cos(2 * 3.14159 * Y);
}



// 画像にノイズを加える
void add_noise_8U(Mat& img, Mat& out, float sigma) {
	img.copyTo(out);
	// ガウスノイズを加える
	for (int ptr = 0; ptr < img.cols * img.rows; ptr++) {
		int v = img.data[ptr] + Box_Muller() * sigma;
		if (v < 0)v = 0;
		if (v > 255)v = 255;
		out.data[ptr] = v;
	}
}

void gen_blob_ellipse_kadai(Mat& img, int seed) {

	//************************************************************
	// change seed number by last four digits of your ID
	srand(seed);

	img = Mat(512, 512, CV_8U);

	img = 20;

	for (int i = 0; i < 30; i++) {
		float scale = 0.1 + (rand() % 80 + 20) / 100.0;
		ellipse(img, Point(rand() % 512, rand() % 512), Size(10 * scale, 20 * scale), rand() % 360, 0, 360, 140 + rand() % 100, -1, LINE_AA);
	}
	add_noise_8U(img, img, 20.0);

	imshow("img", img);
}





void test_blob_opencv() {

	// gen blob image
	Mat img;
	gen_blob_ellipse_kadai(img, 8746);

	imshow("img", img);
	imwrite("original.png", img);

	// binalize
	Mat bin;
	int T;
	T = threshold(img, bin, 0, 255, THRESH_OTSU);
	printf("T=%d\n", T);

	imshow("bin", bin);
	imwrite("bin.png", bin);
	// ノイズ除去
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));

	// クロージング 膨張→収縮
	Mat dilation_dst;
	dilate(bin, dilation_dst, kernel);
	imshow("dilation_dst", dilation_dst);
	Mat closing_dst;
	erode(dilation_dst, closing_dst, kernel);
	imshow("closing_dst", closing_dst);

	// クロージング処理した画像にオープニング処理 収縮→膨張
	Mat erosion_dst;
	erode(closing_dst, erosion_dst, kernel);
	imshow("erosion_dst", erosion_dst);
	Mat opening_dst;
	dilate(erosion_dst, opening_dst, kernel);
	imshow("opening_dst", opening_dst);

	Mat noise_reduction;
	opening_dst.copyTo(noise_reduction);

	imwrite("noise_reduction.png", noise_reduction);






	// labeling
	Mat label;

	Mat stats;
	Mat centroids;

	int max_L = connectedComponentsWithStats(noise_reduction, label, stats, centroids, 4);


	// // coloring by label
	Mat label_rgb = Mat(img.rows, img.cols, CV_8UC3);
	std::vector<Vec3b> lut(max_L);
	for (int i = 0; i < max_L; i++)
		lut[i] = Vec3b(rand() % 255, rand() % 255, rand() % 255);
	label_rgb = 0;
	for (int y = 0; y < label.rows; y++) {
		for (int x = 0; x < label.cols; x++) {
			int L = label.at<int>(y, x);
			if (L > 0) {
				label_rgb.at<Vec3b>(y, x) = lut[L];
			}
		}
	}
	//名前
	Mat out1;
	label_rgb.copyTo(out1);
	String text = "231d-8746";
    cv::putText(out1, text, cv::Point(15,25), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 1);
	imshow("label", out1);
	imwrite("labeling.png", out1);


	// while (1)

	//level2
	// blob解析
	Mat out2;
	label_rgb.copyTo(out2);
	int blob_count2 = 0;
	for (int L = 1; L < max_L; L++) {
		Mat blob = 255 * (label == L);
		// circle(out2, Point(centroids.at<Vec2d>(L,0)), 3, Vec3b(0, 0, 255), -1, LINE_AA);
		// 面積の取得
		int area = stats.at<int>(L, 4);
		printf("L=%d, area=%d\n", L, area);
		// 300pix以上のblobを囲む
		if (area >= 300){		
		Rect rect = stats.at<Rect>(L, 0);
		rectangle(out2, rect, Vec3b(0, 255, 0), 1, LINE_AA);
		blob_count2 ++;
		}
		printf("counts=%d\n", blob_count2);
	}
	
	String text_n2 = "231d-8746, N = " + to_string(blob_count2);
	cv::putText(out2, text_n2, cv::Point(15,25), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 1);
	imshow("blob", out2);
	imwrite("blob2.png", out2);





	// level3
	Mat out3;
	label_rgb.copyTo(out3);
	int blob_count = 0;
	for (int L = 1; L < max_L; L++) {
		Mat blob = 255 * (label == L);
		
		// 面積の取得
		int area = stats.at<int>(L, 4);
		printf("L=%d, area=%d\n", L, area);
		// 300pix以上のblobを囲む
		if (area >= 300){		
		
		Rect rect = stats.at<Rect>(L, 0);
		
		Moments moments = cv::moments(blob, true);
		double mu20 = moments.mu20;
        double mu02 = moments.mu02;
        double mu11 = moments.mu11;
		
		// θの計算
		double angle = 0.5 * std::atan2(2*mu11, mu20-mu02);
		// //度に変換
		// double convert_angle = angle * 180 / CV_PI;
		



		// 重心
        double cx = centroids.at<double>(L, 0);
        double cy = centroids.at<double>(L, 1);

		Point center(static_cast<int>(cx), static_cast<int>(cy));
        Point end;
        end.x = center.x + static_cast<int>(30 * std::cos(angle));
        end.y = center.y + static_cast<int>(30 * std::sin(angle));

        line(out3, center, end, cv::Scalar(0, 0, 255), 1);

		rectangle(out3, rect, Vec3b(0, 255, 0), 1, LINE_AA);
		blob_count ++;
		}
		printf("counts=%d\n", blob_count);
	}
	
	String text_n = "231d-8746, N = " + to_string(blob_count);
	cv::putText(out3, text_n, cv::Point(15,25), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 1);
	imshow("moment", out3);
	imwrite("moment.png", out3);
	waitKey(0);
}


int main() {

	test_blob_opencv();

}
