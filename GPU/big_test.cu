#include <cstdio>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iterator>
#include <vector>
#include "neural_network.cuh"

using namespace std;

int main()
{
	vector<pair<vector<float>, vector<float> > > tests;
	ifstream train_stream("train-set");
	string line;
	while (getline(train_stream, line))
	{
		vector<float> test;
		vector<float> ans;
		float value;
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
 	float learning_speed = 0.01;
 	float learning_add = 0.005;
	float momentum = 0.1f;
	float momentum_sub = 0.01;
	float alpha = 1.7159;
	float beta = 2.0 / 3.0;
	float lambda = 0.3;

	float error = 0.004f;
	int max_iterations = 10;

	//float max_val = 5;
	//float min_freq = 1500;

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
	vector<float> p(2);
	vector<float> r(2);
	vector<float> f1(2);
	while (getline(test_stream, line))
	{
		vector<float> test;
		vector<float> ans;
		float value;
		istringstream iss(line);
		iss >> value;
		ans.push_back(value);
		sum[(int)value]++;	
		iss >> value;
		while (iss >> value)
		{
			test.push_back(value);
		}
		vector<float> net_ans = net.calculate(test);
		float round_ans = floor(net_ans[0] + 0.5f);
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
	float tests_passed = 100.0f * (test_id - error_count) / test_id;
	printf("\nerrors: %d; %.2f percent of tests passed\n\n", error_count, tests_passed);
	for (int i = 0; i < 2; i++)
	{
		p[i] = 1.0f * sum_net_correct[i] / sum_net[i];
		r[i] = 1.0f * sum_net_correct[i] / sum[i];
		f1[i] = 1.0f * ((2.0f * p[i] * r[i]) / (p[i] + r[i]));
	}
	printf("p[0]: %f r[0]: %f \n", p[0], r[0]);
	printf("p[1]: %f r[1]: %f \n", p[1], r[1]);
	printf("f1[0]: %f f1[1]: %f\n", f1[0], f1[1]);
	float p_avg = 1.0 * (p[0] + p[1]) / 2;
	float r_avg = 1.0 * (r[0] + r[1]) / 2;
	float f1_avg = (2.0 * r_avg * p_avg) / (r_avg + p_avg);
	printf("\nf1_avg: %.5f", f1_avg);
	return 0;
}