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

struct obj_t {
	int     idx;
	Point2f pt;
};

// テンプレートの作成
void gen_template(vector<Mat> & Ts, char *name) {

	for (int i = 0; i < strlen(name); i++) {
		Mat T = Mat(32, 32, CV_8U);
		T = 0;
		char buf[32];
		sprintf(buf, "%c", name[i]);
		putText(T, buf, Point(8, 24), FONT_HERSHEY_SIMPLEX, 0.7, 255, 2.0, LINE_AA);
        imwrite("template"+ to_string(i) +".png", T);
		Ts.push_back(T);
	}
}

void gen_input_subpix_image(Mat & I, vector<Mat>& Ts, vector<obj_t>& objs){




}

void gen_input_image(Mat & I, vector<Mat>& Ts, vector<obj_t>& objs) {

	int w, h;
	w = h = 512;
	I = Mat(h, w, CV_8U);
	I = 0;
	// paste random object & position
	for (int i = 0; i < 20; i++) {
		obj_t obj;
		obj.idx = rand()%Ts.size();
		obj.pt.x = (w- Ts[obj.idx].cols) * (rand() % 10000) / 10000.0;
		obj.pt.y = 32 + (h- Ts[obj.idx].rows - 32) * (rand() % 10000) / 10000.0;
		objs.push_back(obj);
		Mat A = (cv::Mat_<double>(2, 3) << 1, 0, obj.pt.x, 0, 1, obj.pt.y);
		Mat tmp;
		cv::warpAffine(Ts[obj.idx], tmp, A, I.size());
		I = I | tmp;
	}
}
//ZNCC
void detect(Mat& I, Mat & RGB, vector<Mat>& Ts, vector<obj_t>& res) {



	for (int i = 0; i < Ts.size(); i++) {

		Mat sim;

		matchTemplate(I, Ts[i], sim, cv::TM_CCOEFF_NORMED);

		for (int y = 1; y < sim.rows -1; y++) {
			for (int x = 1; x < sim.cols-1; x++) {
                // 注目画素
				float p0 = sim.at<float>(y, x);
                
                
				// detect by threshold
				if (p0 > 0.8) {
					// add result
                    //注目画像が閾値を超えていた場合．
					obj_t obj;
					obj.pt.x = x;
					obj.pt.y = y;
					obj.idx = i;
					res.push_back(obj);
					// printf("detect %d (%d, %d)\n", i, x, y);

					// Level 1
					// draw rectangle & ID
                    String text = "ID=" + to_string(i); 
                    putText(RGB, text, Point(obj.pt.x, obj.pt.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
                    rectangle(RGB, Rect(Point(obj.pt.x, obj.pt.y), Point(obj.pt.x, obj.pt.y) + Point(Ts[i].cols, Ts[i].rows)), CV_RGB(255, 0, 0), 1.0, LINE_AA);
                    
				}
			}
		}

	}
}

//ZNCC
void peek_detect(Mat& I, Mat & RGB, vector<Mat>& Ts, vector<obj_t>& res) {



	for (int i = 0; i < Ts.size(); i++) {

		Mat sim;

		matchTemplate(I, Ts[i], sim, cv::TM_CCOEFF_NORMED);

		for (int y = 1; y < sim.rows -1; y++) {
			for (int x = 1; x < sim.cols-1; x++) {
                // 注目画素
				float p5 = sim.at<float>(y, x);
                // 8近傍
				float p1 = sim.at<float>(y-1, x-1);
				float p2 = sim.at<float>(y-1, x);
				float p3 = sim.at<float>(y-1, x+1);
				float p4 = sim.at<float>(y, x-1);
                float p6 = sim.at<float>(y, x+1);
                float p7 = sim.at<float>(y+1, x-1);
                float p8 = sim.at<float>(y+1, x);
                float p9 = sim.at<float>(y+1, x+1);

                float max_value = p5;
                int max_index = 0;


                float values[] = {p1, p2, p3, p4, p6, p7, p8, p9};

                for (int j = 0; j < 8; j++){
                    if (values[j] > max_value){
                        max_value = values[j];
                        max_index = j + 1;
                    }
                }
                
				// detect by threshold
				if (max_index == 0 && p5 > 0.8) {
                    // printf("%f\n", p5);
					// add result
                    //注目画像が閾値を超えていた場合．
					obj_t obj;
					obj.pt.x = x;
					obj.pt.y = y;
					obj.idx = i;
					res.push_back(obj);
     
					printf("detect %d (%d, %d)\n", i, obj.pt.x, obj.pt.y);

					// Level 1
					// draw rectangle & ID
                    String text = "ID=" + to_string(i); 
                    putText(RGB, text, Point(obj.pt.x, obj.pt.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
                    rectangle(RGB, Rect(Point(obj.pt.x, obj.pt.y), Point(obj.pt.x, obj.pt.y) + Point(Ts[i].cols, Ts[i].rows)), CV_RGB(255, 0, 0), 1.0, LINE_AA);
                    
				}
			}
		}

	}
}
void eval(vector<obj_t>& objs, vector<obj_t>& res, int& obj_num, float& prec, float& total_d, int& correct_count){

    
    obj_num = objs.size();
    int res_num = res.size();
    int detect_count = 0;
    total_d = 0;
    correct_count = 0;
   for (size_t i = 0; i < obj_num; i++) {
        obj_t obj = objs[i];
        bool detected  = false;
        int obj_count = 0;
        float d = 0;


        for (int j = 0; j < res_num; j++){
            obj_t re = res[j];
            // idxにはそれぞれの文字を示すLabelが入っている．
            if (obj.idx == re.idx){
            float x_d = std::abs(obj.pt.x - re.pt.x);
            float y_d = std::abs(obj.pt.y - re.pt.y);
            d = std::sqrt(x_d*x_d+y_d*y_d);
            if (d <= 3.0){
                detected = true;
                obj_count += 1;
                // break;
            }
            if (detected){
            detect_count += 1;
            total_d += d;
            detected = false;
        }
            }

        }
        if (obj_count > 0){
            correct_count += 1;
        }
        
    }
    
    if (detect_count != 0){
        prec = 100*correct_count / obj_num;
        total_d /= detect_count;
    }
 

}


void sub_pix_2D(float p1, float p2, float p3, float p4, float p5, float p6, float p7, float p8, float p9, float& sub_x, float& sub_y){

    float b1, b2, b3, b4, b5, b6;
    float a, b, c, d, e, f;
    
    b1 = p1 + p3 + p4 + p6 + p7 + p9;
    b2 = p1 - p3 - p7 + p9;
    b3 = p1 + p2 + p3 + p7 + p8 + p9;
    b4 = -p1 + p3 - p4 + p6 - p7 + p9;
    b5 = -p1 -p2 -p3 + p7 + p8 + p9;
    b6 = p1 + p2 + p3 + p4 + p5 +p6 + p7 + p8 + p9;

    a = (3*b1 - 2*b6)/6.0;
    b = b2 / 4.0;
    c = (3*b3-2*b6) / 6.0;
    d = b4 / 6.0;
    e = b5 / 6.0;
    f = (-3*b1-3*b3-5*b6) / 9.0;

    sub_x = (2*c*d-b*e)/(b*b-4*a*c);
    sub_y = (2*a*e-b*d)/(b*b-4*a*c);

    
    // peek = a*sub_x*sub_x + b*sub_x*sub_y + c*sub_y*sub_y + d*sub_x + e*sub_y + f;
    
}
//ZNCC
void detect_subpix(Mat& I, Mat & RGB, vector<Mat>& Ts, vector<obj_t>& res) {



	for (int i = 0; i < Ts.size(); i++) {

		Mat sim;

		matchTemplate(I, Ts[i], sim, cv::TM_CCOEFF_NORMED);

		for (int y = 1; y < sim.rows -1; y++) {
			for (int x = 1; x < sim.cols-1; x++) {
                // 注目画素
				float p5 = sim.at<float>(y, x);
                // 8近傍
				float p1 = sim.at<float>(y-1, x-1);
				float p2 = sim.at<float>(y-1, x);
				float p3 = sim.at<float>(y-1, x+1);
				float p4 = sim.at<float>(y, x-1);
                float p6 = sim.at<float>(y, x+1);
                float p7 = sim.at<float>(y+1, x-1);
                float p8 = sim.at<float>(y+1, x);
                float p9 = sim.at<float>(y+1, x+1);

                

                float max_value = p5;
                int max_index = 0;


                float values[] = {p1, p2, p3, p4, p6, p7, p8, p9};

                for (int j = 0; j < 8; j++){
                    if (values[j] > max_value){
                        max_value = values[j];
                        max_index = j + 1;
                    }
                }

				// detect by threshold
				if (p5 > 0.8 && max_index == 0) {

					// add result
                    //注目画像が閾値を超えていた場合．
					obj_t obj;
                    //サブピクセル補正．
                    float sub_x, sub_y;
                    sub_pix_2D(p1, p2, p3, p4, p5, p6, p7, p8, p9, sub_x, sub_y);
					obj.pt.x = sub_x + x;
					obj.pt.y = sub_y + y;
					obj.idx = i;
					res.push_back(obj);
                    printf("detect %d (%f, %f)\n", i, obj.pt.x, obj.pt.y);

	
                    String text = "ID=" + to_string(i); 
                    putText(RGB, text, Point(obj.pt.x, obj.pt.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
                    rectangle(RGB, Rect(Point(obj.pt.x, obj.pt.y), Point(obj.pt.x, obj.pt.y) + Point(Ts[i].cols, Ts[i].rows)), CV_RGB(255, 0, 0), 1.0, LINE_AA);
                    
				}
			}
		}

	}
}
int main() {

	// gen tamplates: change your name with avoiding duplication. 
	// ex) GouKoutaki -> GouKtaki
	std::vector<Mat> Ts;
	gen_template(Ts, "YoshinagAtu");

	// gen input image
	Mat I;
	vector<obj_t> objs;
	gen_input_image(I, Ts, objs);
	imshow("Input", I);
    imwrite("in.png", I);

	// run detection
	vector<obj_t> res1;
	Mat out1;
	cvtColor(I, out1, COLOR_GRAY2BGR);
	detect(I, out1, Ts, res1);
    imshow("detect", out1);
    imwrite("detect.png",out1);
	// Level2 & Level3 : Add Evaluation

    //peek値処理を加えた場合
    vector<obj_t> res2;
	Mat out2;
	cvtColor(I, out2, COLOR_GRAY2BGR);
	peek_detect(I, out2, Ts, res2);
	// Level2 & Level3 : Add Evaluation
    int correct_count2, obj_num2;
    float prec2, total_d2;
    // 検出率と検出成功の場合の誤差の平均．
    eval(objs, res2, obj_num2, prec2, total_d2, correct_count2);
    String eval_text2 = to_string(correct_count2) + "/" + to_string(obj_num2) + "(prec = " + to_string(prec2) + "%), mean_d = " + to_string(total_d2);
    cv::putText(out2, eval_text2, cv::Point(15,20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

    // eval(objs, res2, obj_num, prec, total_d, detect_count);

    // String eval_text = to_string(detect_count) + "/" + to_string(obj_num) + "(prec = " to_string(prec) + "%), mean_d = " + to_string(total_d);
    imshow("detect2", out2);
    imwrite("peek_detect.png",out2);


    vector<obj_t> res3;
	Mat out3;
	cvtColor(I, out3, COLOR_GRAY2BGR);
	detect_subpix(I, out3, Ts, res3);
    int correct_count3, obj_num3;
    float prec3, total_d3;
    // 検出率と検出成功の場合の誤差の平均．
    eval(objs, res3, obj_num3, prec3, total_d3, correct_count3);
    String eval_text3 = to_string(correct_count3) + "/" + to_string(obj_num3) + "(prec = " + to_string(prec3) + "%), mean_d = " + to_string(total_d3);
    cv::putText(out3, eval_text3, cv::Point(15,20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    imshow("detect3", out3);
    imwrite("subpix_detect.png",out3);
    waitKey(0);

}



