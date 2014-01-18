#include <vector>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iterator>
#include <math.h>
#include "neural_network.h"

using namespace std;

int main()
{
	vector<pair<vector<double>, vector<double> > > tests;
	ifstream train_stream("train-set");
	string line;
	while (getline(train_stream, line))
	{
		vector<double> test;
		vector<double> ans;
		double value;
		istringstream iss(line);
		iss >> value;
		ans.push_back(value);
		iss >> value;
		while (iss >> value)
		{
			test.push_back(value);
		}
		tests.push_back(make_pair(test, ans));
	}

	int inputs = 141;
	int outputs = 1;
	int depth = 3;
	//int hidden_layer_size = 4;
	int hidden_layer_size = 20;
	//double learning_speed = 0.9;
	//double momentum = 0.6;
	double learning_speed = 0.1;
	double momentum = 0.1;
	//double alpha = 0.001;
	double alpha = 1;

	double error = 0.004;
	int max_iterations = 1000;

	double max_val = 5;
	double min_freq = 1500;

	//double error = 0.08;
	//int max_iterations = 2000;

	neural_network net(inputs, depth, hidden_layer_size, outputs, learning_speed, momentum, alpha);

	net.teach(tests, error, max_iterations, max_val, min_freq);

	freopen("results", "w", stdout);
	ifstream test_stream("test-set");
	int test_id = 1;
	int error_count = 0;
	while (getline(test_stream, line))
	{
		vector<double> test;
		vector<double> ans;
		double value;
		istringstream iss(line);
		iss >> value;
		ans.push_back(value);
		iss >> value;
		while (iss >> value)
		{
			test.push_back(value);
		}
		vector<double> net_ans = net.calculate(test);
		double round_ans = floor(net_ans[0] + 0.5);
		if ((int)round_ans != (int)ans[0])
		{
			printf("ERROR! test id: %d correct: %d net_output: %f\n", test_id, (int)ans[0], net_ans[0]);
			error_count++;
		}
		else
		{
			printf("OK! test id: %d correct: %d net_output: %f\n", test_id, (int)ans[0], net_ans[0]);
		}
		test_id++;
	}
	double tests_passed = 100.0 * (test_id - error_count) / test_id;
	printf("\nerrors: %d; %.2f percent of tests passed", error_count, tests_passed);

	return 0;
}