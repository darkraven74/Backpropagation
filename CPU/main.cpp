#include <vector>
#include <stdio.h>
#include "neural_network.h"

int main()
{
	int inputs = 2;
	int outputs = 1;
	int depth = 3;
	int hidden_layer_size = 2;
	double learning_speed = 0.1;
	double momentum = 0.9;
	double error = 0.001;

	neural_network net(inputs, depth, hidden_layer_size, outputs, learning_speed, momentum);
	vector<pair<vector<double>, vector<double> > > tests;
	vector<double> test(2);
	vector<double> ans(1);
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			test[0] = i;
			test[1] = j;
			ans[0] = (i + j) % 2;
			tests.push_back(make_pair(test, ans));
		}
	}
	net.teach(tests, error);
	for (int i = 0; i < tests.size(); i++)
	{
		vector<double> ans = net.calculate(tests[i].first);
		printf("input: %d %d\n", (int)tests[i].first[0], (int)tests[i].first[1]);
		printf("output: %f\n\n", ans[0]);
	}

	return 0;
}