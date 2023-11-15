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
// 画像の読み込み．
void readImages(vector<Mat>& images) {
	for (int i = 0; i < 100; i++) {
		char file[1024];
		sprintf(file, "./faces/%06d.jpg", i + 1);
		// load as gray image
		Mat img = imread(file, 0);
		if (img.data)
		{
			img.convertTo(img, CV_32F, 1 / 255.0);
			images.push_back(img);
			// flip and add
			Mat imgFlip;
			flip(img, imgFlip, 1);
			images.push_back(imgFlip);
		}
	}
}
// Create data matrix from a vector of images
Mat createDataMatrix(const vector<Mat>& images)
{
	// data(N x pix)
	Mat data(images.size(), images[0].rows * images[0].cols, CV_32F);
	for (int i = 0; i < images.size(); i++)
	{
		Mat image = images[i].reshape(1, 1);
		image.copyTo(data.row(i));
	}
	return data;
}
// 平均顔の計算．
void compute_eigenface(vector<Mat> & images, Mat& averageFace, vector<Mat>& eigenFaces) {
	Size sz = images[0].size();

	Mat data = createDataMatrix(images);
	int N = images.size();

	// compute PCA
	
	PCA pca(data, Mat(), PCA::DATA_AS_ROW, N);

	// average face
	averageFace = pca.mean.reshape(1, sz.height);
	// 保存用
	Mat tmp_averageFace;
	averageFace.copyTo(tmp_averageFace);

	// 平均顔の保存．
	// normalize(tmp_averageFace, tmp_averageFace, 0, 255, NORM_MINMAX);
	tmp_averageFace.convertTo(tmp_averageFace, CV_8U, 255);
	imwrite("average_face.png", tmp_averageFace);

	// Find eigen vectors.
	Mat eigenVectors = pca.eigenvectors;

	// Reshape Eigenvectors to obtain EigenFaces
	for (int i = 0; i < N; i++)
	{
		Mat eigenFace = eigenVectors.row(i).reshape(1, sz.height);
		eigenFaces.push_back(eigenFace);
	}

	for (int i = 0; i < eigenFaces.size(); ++i) {

		// Convert eigen face to CV_8U
		Mat eigenface;
		eigenFaces[i].copyTo(eigenface);
		normalize(eigenface, eigenface, 0, 255, NORM_MINMAX);
    	eigenface.convertTo(eigenface, CV_8U);
        string filename = "eigenface_" + to_string(i) + ".png";
        imwrite(filename, eigenface);
    }



    // waitKey(0);

}




// データセットからランダムに選択した画像を貼り付ける
void gen_face(vector<Mat> images, Mat & img, vector<int> & labels, int seed) {

	srand(seed);
	img = Mat(800, 1200, CV_32F);
	img = 1;
	int w = images[0].cols, h = images[0].rows;

	

	for (int y = 50; y <= 700; y += 250) {
		for (int x = 50; x <= 1000; x += 200) {
			int id = rand() % images.size();
			Mat roi(img, Rect(x, y, w, h));
			images[id].copyTo(roi);
			labels.push_back(id);

		}
	}
	img.convertTo(img, CV_8U, 255);
	imshow("img", img);
	// waitKey(0);
}


// sort rect according to position (left, top) : TV scan
bool fcomp(const Rect& a, const Rect& b) { return (a.y + a.x*0.1) < (b.y + b.x * 0.1); }

int d_labels_index(int face_x, int face_y, vector<int> labels){

	int index = 0;
	for (int y = 50; y <= 700; y += 250) {
		for (int x = 50; x <= 1000; x += 200) {

			if(face_y >= y && face_y <= y+250 && face_x >= x && face_x <= x+200){
				return index;
			}

		index++;

		}
	}

	return index;


}
// level 3
void face_detect_crop(Mat & img, vector<Mat> & detect_faces, vector<int>  labels, vector<int> & d_labels) {

	CascadeClassifier cascade;
	cascade.load("haarcascade_frontalface_alt.xml");
	vector<Rect> faces;
	// resize用
	int resize_w = 178, resize_h = 218;
	// 顔周辺を切り出すためのマージン
	const int margin = 30;

	// run detect
	cascade.detectMultiScale(img, faces, 1.1, 3, 0, Size(20, 20));

	sort(faces.begin(), faces.end(), fcomp);
	Mat rgb;
	cvtColor(img, rgb, COLOR_GRAY2BGR);
	for (int i = 0; i < faces.size(); i++)
	{
		int w = faces[i].width, h = faces[i].height;
		rectangle(rgb, Rect(faces[i].x, faces[i].y, w, h), Scalar(0, 0, 255), 3, LINE_AA);
		// crop head regeion
		    int x = max(0, faces[i].x - margin);
			int y = max(0, faces[i].y - margin);
			int dw = min(img.cols - x, faces[i].width + 2 * margin);
			int dh = min(img.rows - y, faces[i].height + 2 * margin);
            Rect rect(x, y, dw, dh);
			// Rect rect(faces[i].x, faces[i].y, w, h);

		Mat roi(img, rect), tmp;
		roi.copyTo(tmp);
		resize(tmp, tmp, Size(resize_w, resize_h));
		imwrite("detect" + to_string(i) +".png", tmp);
		detect_faces.push_back(tmp);
		int d_label_index = d_labels_index(faces[i].x, faces[i].y, labels);
		d_labels.push_back(labels[d_label_index]);

		
	}

	imshow("detect face", rgb);
    imwrite("detect.png", rgb);
	// waitKey(0);
}
// 最近傍法

int findNearestNeighbor(const cv::Mat& query, const cv::Mat& database) {
    int nearestIndex = -1;
    double minDistance = std::numeric_limits<double>::infinity();

    for (int i = 0; i < database.rows; ++i) {
        double distance = cv::norm(query, database.row(i), cv::NORM_L2);

        if (distance < minDistance) {
            minDistance = distance;
            nearestIndex = i;
        }
    }

    return nearestIndex;
}

// level max
void test_recognition(vector<Mat> & detect_faces, vector<Mat> & images, vector<int> labels, Mat &averageFace, vector<Mat> & eigenFaces) {

	int M = 40;	// low-rank dimension
	int N = images.size();
	Size sz = images[0].size();
	int correct_counts = 0;
	int correct_counts2 = 0;
	// gen dictionary of face database images
	Mat dict(N, M, CV_64F);
	// 元画像に関して
	// ai = (x - μ)piこれを全部の画像で計算．
	for (int i = 0; i < N; i++) {
		for (int m = 0; m < M; m++) {
			dict.at<double>(i, m) = (images[i] - averageFace).dot(eigenFaces[m]);
		}
	}

    for (int i = 0; i < detect_faces.size(); i++) {

	Mat target;

	images[labels[i]].convertTo(target, CV_32F);
	resize(target, target, averageFace.size());

	Mat reconst;
	imshow("target", target);

	// // reconstruction
	averageFace.copyTo(reconst);
	Mat q(1, M, CV_64F);
	for (int m = 0; m < M; m++) {
		double a = (target - averageFace).dot(eigenFaces[m]);

		reconst += a * eigenFaces[m];
		q.at<double>(m) = a;
	}
	imshow("reconst", reconst);
	imwrite("reconst_" + to_string(i) + ".png", reconst*255);

	//*********************************************************
	// find nearest neighbor from database dict for query q
	int detect_id = findNearestNeighbor(q, dict);
	printf("detect_id=%d, correct_id=%d\n", detect_id, labels[i]);
	if (detect_id == labels[i]){
		correct_counts += 1;
	}
	} 
	printf("元画像\n");
	float recognition_rate =  100*static_cast<float>(correct_counts) / detect_faces.size();
	printf("recognition_rate:%f%%\n", recognition_rate);


	// 検出画像の復元
	for (int i = 0; i < detect_faces.size(); i++){
	Mat target2;
	detect_faces[i].convertTo(target2, CV_32F, 1/255.0);
	resize(target2, target2, averageFace.size());

	Mat reconst2;
	averageFace.copyTo(reconst2);

	Mat q2(1, M, CV_64F);
	for (int m = 0; m < M; m++) {
		double a = (target2 - averageFace).dot(eigenFaces[m]);

		reconst2 += a * eigenFaces[m];
		q2.at<double>(m) = a;
	}
	imshow("reconst2", reconst2);
	imwrite("reconst2_" + to_string(i) + ".png", reconst2*255);

	//*********************************************************
	// find nearest neighbor from database dict for query q
	int detect_id = findNearestNeighbor(q2, dict);
	printf("detect_id=%d, correct_id=%d\n", detect_id, labels[i]);
	if (detect_id == labels[i]){
		correct_counts2 += 1;
	}
	}
	float recognition_rate2 =  100*static_cast<float>(correct_counts2) / detect_faces.size();
	printf("検出切り出し画像\n");
	printf("recognition_rate:%f%%\n", recognition_rate2);
// 	printf("recognition_rate: %f", correct_counts / detect_faces.size());
}


int main() {

	Mat img;
	// load database face
	vector<Mat> images;
	readImages(images);

	// compute eigen faces 固有顔画像 PCAの結果
	Mat averageFace;
	vector<Mat> eigenFaces;
	compute_eigenface(images, averageFace, eigenFaces);

	// gen test input image
	vector<int> labels;

	gen_face(images, img, labels, 8746);

    // // imgに対して，顔検出を行う．
	// face detect(level1) & crop faces : level 3
	vector<Mat> detect_faces;
	vector<int> d_labels;
	face_detect_crop(img, detect_faces, labels, d_labels);

	// // run recognition：level max
	test_recognition(detect_faces, images, d_labels, averageFace, eigenFaces);

    waitKey(0);

}



