#include "neural_network.h"
#include <vector>

int main()
{
	neural_network net(2, 3, 2, 1);
	vector<double> test(2, 1);
	vector<double> ans(1, 0);
	pair<vector<double>, vector<double> > p = make_pair(test, ans);
	vector<pair<vector<double>, vector<double> > > tests;
	tests.push_back(p);
	net.teach(tests);
	return 0;
}