#include <cstdio>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iterator>
#include <vector>
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
	int depth = 5;
	int hidden_layer_size = 100;
	double learning_speed = 0.01;
	double learning_add = 0.005;
	double momentum = 0.1;
	double momentum_sub = 0.01;
	double alpha = 1.7159;
	double beta = 2.0 / 3.0;
	double lambda = 0.3;
	//double lambda = 0.5;

	double error = 0.0001;
	//double error = 0.025;
	int max_iterations = 10;

	//double max_val = 5;
	//double min_freq = 1500;

	neural_network net(inputs, depth, hidden_layer_size, outputs, learning_speed, learning_add, momentum, 
		momentum_sub, alpha, beta, lambda);
	//neural_network net("net.txt");

	net.teach(tests, error, max_iterations);
	//net.save_to_file("net.txt");

	freopen("results", "w", stdout);
	ifstream test_stream("test-set");
	int test_id = 1;
	int error_count = 0;
	vector<int> sum(2);
	vector<int> sum_net(2);
	vector<int> sum_net_correct(2);
	vector<double> p(2);
	vector<double> r(2);
	vector<double> f1(2);

	while (getline(test_stream, line))
	{
		vector<double> test;
		vector<double> ans;
		double value;
		istringstream iss(line);
		iss >> value;
		ans.push_back(value);
		sum[(int)value]++;	
		iss >> value;
		while (iss >> value)
		{
			test.push_back(value);
		}
		vector<double> net_ans = net.calculate(test);
		double round_ans = floor(net_ans[0] + 0.5);
		if (round_ans < 0.0)
		{
			round_ans = 0;
		}
		if (round_ans > 1.0)
		{
			round_ans = 1;
		}
		sum_net[(int)round_ans]++;	
				
		if ((int)round_ans != (int)ans[0])
		{
			//printf("ERROR! test id: %d correct: %d net_output: %f\n", test_id, (int)ans[0], net_ans[0]);
			error_count++;
		}
		else
		{
			sum_net_correct[(int)round_ans]++;
			//printf("OK! test id: %d correct: %d net_output: %f\n", test_id, (int)ans[0], net_ans[0]);
		}
		test_id++;
	}
	double tests_passed = 100.0 * (test_id - error_count) / test_id;
	char* percent = "%";
	printf("\nerrors: %d; %.2f%s tests passed ", error_count, tests_passed, percent);
	for (int i = 0; i < 2; i++)
	{
		p[i] = 1.0 * sum_net_correct[i] / sum_net[i];
		r[i] = 1.0 * sum_net_correct[i] / sum[i];
		f1[i] = 1.0 * ((2.0 * p[i] * r[i]) / (p[i] + r[i]));
	}
	printf("p[0]: %.3f r[0]: %.3f             ", p[0], r[0]);
	printf("p[1]: %.3f r[1]: %.3f             ", p[1], r[1]);
	printf("f1[0]: %.3f f1[1]: %.3f", f1[0], f1[1]);
	double p_avg = 1.0 * (p[0] + p[1]) / 2;
	double r_avg = 1.0 * (r[0] + r[1]) / 2;
	double f1_avg = (2.0 * r_avg * p_avg) / (r_avg + p_avg);
	printf("\nf1_avg: %.5f", f1_avg);
	return 0;
}