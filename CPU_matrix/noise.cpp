#include <cstdio>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <sstream>
#include <iterator>
#include <vector>
#include <algorithm>

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

	srand(time(NULL));
	random_shuffle(tests.begin(), tests.end());
	int count = tests.size();
	count *= 0.01;
	double min_w = -0.5;
	double max_w = 0.5;
	for (int i = 0; i < count; i++)
	{
		int id = rand() % tests.size();
		vector<double> test = tests[id].first;
		vector<double> ans = tests[id].second;
		for (int j = 0; j < test.size(); j++)
		{
			test[j] += ((max_w - min_w) * ((double)rand() / (double)RAND_MAX) + min_w);
			//test[j] = rand() % 5;
		}
		/*if (rand() % 2)
		{
			ans[0]++;
			ans[0] = (int)ans[0] % 2;
		}*/
		tests.push_back(make_pair(test, ans));
	}
	random_shuffle(tests.begin(), tests.end());
	freopen("train-set-noise", "w", stdout);
	for (int i = 0; i < tests.size(); i++)
	{
		printf("%d 0 ", (int)tests[i].second[0]);
		for (int j = 0; j < tests[i].first.size(); j++)
		{
			printf("%f ", tests[i].first[j]);
		}
		printf("\n");
	}
	
	return 0;
}