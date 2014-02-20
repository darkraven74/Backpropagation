#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <string>

struct layer
{
	layer(int size, int inputs);
	double* weights;
	double* delta_weights;
	double* outputs;
	double* borders;
	double* deltas;
	int inputs;
	int size;
};

class neural_network
{
public:
	neural_network(int inputs, int depth, int hidden_layer_size, int outputs, double learning_speed = 0.5,
		double momentum = 0.5, double alpha = 1.0);
	neural_network(std::string file_name);
	void teach(std::vector<std::pair <std::vector<double>, std::vector<double> > >& tests,
		double error, int max_iterations, double max_val, double min_freq);
	std::vector<double> calculate(std::vector<double> const& input);
	void save_to_file(std::string file_name);


private:
	void init();
	void forward_pass(std::vector<double> const& test);
	void backward_pass(std::vector<double> const& test_anwser);
	void normalize(std::vector<std::pair <std::vector<double>, std::vector<double> > > const& tests, double max_val, double min_freq);

	std::vector<double> coeff;
	std::vector<layer> layers;
	int inputs;
	int outputs;
	int depth;
	int hidden_layer_size;
	double learning_speed;
	double momentum;
	double test_error;
	double alpha;
};

#endif // NEURAL_NETWORK_H
