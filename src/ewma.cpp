#include <opencv2/opencv.hpp>

#pragma warning(disable:4996)
#include <iostream>
#include <vector>
#include <chrono>
#include <string>
using std::vector;
using std::string;

#include <opencv2/opencv.hpp>
using cv::Mat;

#include <iostream>
#include <vector>
using std::vector;
using std::chrono::system_clock;
using std::chrono::duration;

#define CVPLOT_HEADER_ONLY
#include "CvPlot/cvplot.h"

#define DTYPE float
#define DEPTH_SCALE 10.F
#define DEPTH_OFFSET -5.F

// Graph attrbs
#define GRAPH_WIDTH 1400
#define GRAPH_HEIGHT 600
#define SPECIFIC_ROW 150

// Exponential Weighted Moving Average properties
#define DEFAULT_ALPHA 0.68
#define DEFAULT_THRESHOLD 5.0 // 5 meter

template <typename T>
class EWMA {
public:
	// Creates the filter
	EWMA();
	EWMA(double alpha);
	EWMA(double alpha, double threshold);

	// Set initial output
	void SetInitialOutput(T output);

	// Specifies a reading value.
	// @returns current output
	const T Filter(T input);

private:
	// Smoothing factor, in range [0, 1.0]
	// Higher the value - less smoothing (higher the latest reading impact).
	double alpha_;

	// Threshold that prevents overshoot
	// Reference: https://dev.intelrealsense.com/docs/depth-post-processing
	double delta_;

	// Current data output(average)
	T output_;
};

template <typename T>
EWMA<T>::EWMA()
	: alpha_(0.6), delta_(8.0), output_(0)
{
}

template <typename T>
EWMA<T>::EWMA(double alpha)
	: delta_(8.0), output_(0)
{
	this->alpha_ = alpha;
}

template <typename T>
EWMA<T>::EWMA(double alpha, double threshold)
	: output_(0)
{
	this->alpha_ = alpha;
	this->delta_ = threshold;
}

template <typename T>
void EWMA<T>::SetInitialOutput(T output)
{
	output_ = output;
}

template <typename T>
const T EWMA<T>::Filter(T input)
{
	// Temporal result
	T tmp = static_cast<T>(alpha_) * input + static_cast<T>(1.0 - alpha_) * output_;

	// Update output_
	if (tmp - output_ < delta_) {
		output_ = tmp;
	}
	else {
		output_ = input;
	}

	return output_;
}

class Timer {
private:
	system_clock::time_point start;
public:
	Timer() { tic(); }
	void tic() {
		start = system_clock::now();
	}

	double toc() {
		duration<double> elapsed = system_clock::now() - start;
		return elapsed.count();
	}
};

void getBinaryData(std::vector<DTYPE>& vec, const char* data_path)
{
	FILE* fp = fopen(data_path, "rb");
	int32_t count = vec.size();
	if (data_path != nullptr) {
		fread(&vec[0], sizeof(DTYPE), count, fp);
	}
	fclose(fp);
}

void convertDisparityToDepth(Mat &disparityMat, Mat &depthMat)
{
	if (depthMat.empty())
	{
		depthMat = Mat(disparityMat.size(), disparityMat.type());
	}
	depthMat = (1.F / disparityMat) * DEPTH_SCALE + DEPTH_OFFSET;
}

void getGraphAtRow(Mat &input, Mat &output, int row)
{
	std::vector<double> row_vec;
	row_vec.resize(input.cols);

	for (int i = 0; i < input.cols; i++)
	{
		row_vec[i] = static_cast<double>(input.ptr<float>(row)[i]);
	}
	auto axes = CvPlot::plot(row_vec, "-");
	output = axes.render(GRAPH_HEIGHT, GRAPH_WIDTH);
}

// in-place filter
void runFilter(EWMA<DTYPE>& ewma, Mat& input_data, uint8_t way = 0)
{
	// way should be one of 0, 1, 2, 3
	// where 0 means positive x axis (default)
	// 1 means positive y axis(as same as OpenCV row direction)
	// 2 means negative x axis
	// 3 means negative y axis
	const int width = input_data.cols;
	const int height = input_data.rows;
	const int channels = input_data.channels();
	std::vector<DTYPE> data;
	data.resize(width * height * channels);
	memcpy(data.data(), input_data.data, sizeof(DTYPE) * data.size());
	if (way == 1) { // transpose can be considered
		for (int col = 0; col < width; col++) { // can be executed independently
			for (int ch = 0; ch < channels; ch++) { // can be executed independently
				ewma.SetInitialOutput(data[channels * (0 * width + col) + ch]);
				for (int row = 1; row < height; row++) {
					DTYPE tmp = data[channels * (row * width + col) + ch];
					data[channels * (row * width + col) + ch] = ewma.Filter(tmp);
				}
			}
		}
	}
	else if (way == 2) {
		for (int row = 0; row < height; row++) { // can be executed independently
			for (int ch = 0; ch < channels; ch++) { // can be executed independently
				ewma.SetInitialOutput(data[channels * (row * width + (width - 1)) + ch]);
				for (int col = width - 2; col >= 0; col--) {
					DTYPE tmp = data[channels * (row * width + col) + ch];
					data[channels * (row * width + col) + ch] = ewma.Filter(tmp);
				}
			}
		}
	}
	else if (way == 3) { // transpose can be considered
		for (int col = 0; col < width; col++) { // can be executed independently
			for (int ch = 0; ch < channels; ch++) { // can be executed independently
				ewma.SetInitialOutput(data[channels * ((height - 1) * width + col) + ch]);
				for (int row = height - 2; row >= 0; row--) {
					DTYPE tmp = data[channels * (row * width + col) + ch];
					data[channels * (row * width + col) + ch] = ewma.Filter(tmp);
				}
			}
		}
	}
	else { // default, include way = 0
		for (int row = 0; row < height; row++) { // can be executed independently
			for (int ch = 0; ch < channels; ch++) { // can be executed independently
				ewma.SetInitialOutput(data[channels * (row * width + 0) + ch]);
				for (int col = 1; col < width; col++) {
					DTYPE tmp = data[channels * (row * width + col) + ch];
					data[channels * (row * width + col) + ch] = ewma.Filter(tmp);
				}
			}
		}
	}
	memcpy(input_data.data, data.data(), sizeof(DTYPE) * data.size());
}

// in-place filter, optimized
void runFilterOpt(EWMA<DTYPE>& ewma, std::vector<DTYPE>& input_data, int width, int height, int channels, uint8_t way = 0)
{
	// way should be one of 0, 1, 2, 3
	// where 0 means positive x axis (default)
	// 1 means positive y axis(as same as OpenCV row direction)
	// 2 means negative x axis
	// 3 means negative y axis
	if (way == 1) { // transpose can be considered
		for (int col = 0; col < width; col++) { // can be executed independently
			for (int ch = 0; ch < channels; ch++) { // can be executed independently
				ewma.SetInitialOutput(input_data[channels * (0 * width + col) + ch]);
				for (int row = 1; row < height; row++) {
					DTYPE tmp = input_data[channels * (row * width + col) + ch];
					input_data[channels * (row * width + col) + ch] = ewma.Filter(tmp);
				}
			}
		}
	}
	else if (way == 2) {
		for (int row = 0; row < height; row++) { // can be executed independently
			for (int ch = 0; ch < channels; ch++) { // can be executed independently
				ewma.SetInitialOutput(input_data[channels * (row * width + (width - 1)) + ch]);
				for (int col = width - 2; col >= 0; col--) {
					DTYPE tmp = input_data[channels * (row * width + col) + ch];
					input_data[channels * (row * width + col) + ch] = ewma.Filter(tmp);
				}
			}
		}
	}
	else if (way == 3) { // transpose can be considered
		for (int col = 0; col < width; col++) { // can be executed independently
			for (int ch = 0; ch < channels; ch++) { // can be executed independently
				ewma.SetInitialOutput(input_data[channels * ((height - 1) * width + col) + ch]);
				for (int row = height - 2; row >= 0; row--) {
					DTYPE tmp = input_data[channels * (row * width + col) + ch];
					input_data[channels * (row * width + col) + ch] = ewma.Filter(tmp);
				}
			}
		}
	}
	else { // default, include way = 0
		for (int row = 0; row < height; row++) { // can be executed independently
			for (int ch = 0; ch < channels; ch++) { // can be executed independently
				ewma.SetInitialOutput(input_data[channels * (row * width + 0) + ch]);
				for (int col = 1; col < width; col++) {
					DTYPE tmp = input_data[channels * (row * width + col) + ch];
					input_data[channels * (row * width + col) + ch] = ewma.Filter(tmp);
				}
			}
		}
	}
}

int main(int argc, char* argv[])
{
	string input_data_str("0.dat");

	if (argc > 1)
	{
		input_data_str = argv[1];
	}

	const int width = 960;
	const int height = 480;

	vector<DTYPE> input_data;
	input_data.resize(width * height);
	getBinaryData(input_data, input_data_str.c_str());

	// input data(as disparity)
	Mat img = Mat(cv::Size(width, height), CV_32F, input_data.data());
	Mat img_depth, img_depth_graph;
	convertDisparityToDepth(img, img_depth); // get final input data(as depth)
	getGraphAtRow(img_depth, img_depth_graph, SPECIFIC_ROW); // get graph

	// show data
	cv::imshow("input_disparity", img); // no need to scale, since tensor has value bet 0 ~ 1.0
	cv::imshow("input_depth", img_depth / 150.0); // since our depth max value is 150m
	cv::imshow("input_depth_graph", img_depth_graph);
	cv::waitKey(10);

	// create EWMA instance
	static EWMA<DTYPE> ewma(DEFAULT_ALPHA, DEFAULT_THRESHOLD);

	vector<DTYPE> input_depth_data;
	input_depth_data.resize(width * height);
	memcpy(input_depth_data.data(), img_depth.data, sizeof(DTYPE) * width * height);

	// Timer instance
	Timer t;
	t.tic();
	// run filter
	bool opt = true;
	for (uint8_t way = 0; way < 4; way++) {
		if (opt) {
			runFilterOpt(ewma, input_depth_data, width, height, 1, way);
		}
		else {
			runFilter(ewma, img_depth, way); // in-place filter that updates original value;
		}
	}
	if(opt) img_depth = Mat(cv::Size(width, height), CV_32F, input_depth_data.data());

	double elapsed = t.toc();
	// print elapsed time
	fprintf(stdout, "Elapsed time: %.4f ms\n", elapsed * 1000.0);

	// show result
	cv::imshow("result_depth", img_depth / 150.0);

	Mat result_depth_graph;
	getGraphAtRow(img_depth, result_depth_graph, SPECIFIC_ROW); // get graph of result
	cv::imshow("result_depth_graph", result_depth_graph);
	cv::waitKey(0);

	return 0;
}