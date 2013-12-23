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
	void teach(vector<pair <vector<double>, vector<double> > > tests);


private:
	void init();
	void forward_pass(vector<double>& test);
	void backward_pass();

	vector<layer> layers;
	int depth;
	int inputs, outputs;
	int hidden_layer_size;
	double learning_speed;
	double momentum;
};

#endif // NEURAL_NETWORK_H
