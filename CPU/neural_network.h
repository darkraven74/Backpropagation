#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>

using namespace std;

struct neuron
{
	neuron(int inputs);
	vector<double> weights;
	vector<double> delta_weights;
	double output;
	double border;
	double delta;
	int inputs;
};

struct layer
{
	layer(int size, int inputs);
	vector<neuron> neurons;
	int size;
};

class neural_network
{
public:
	neural_network(int inputs, int depth, int hidden_layer_size, int outputs);
	neural_network(int inputs, int depth, int hidden_layer_size, int outputs, double learning_speed, double momentum);
	void teach(vector<pair <vector<double>, vector<double> > > tests, double error);
	vector<double> calculate(vector<double> input);


private:
	void init();
	void forward_pass(vector<double>& test);
	void backward_pass(vector<double>& test_anwser);

	vector<layer> layers;
	int inputs;
	int outputs;
	int depth;
	int hidden_layer_size;
	double learning_speed;
	double momentum;
	double test_error;
};

#endif // NEURAL_NETWORK_H
