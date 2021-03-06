#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <string>
#include <random>

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
	neural_network(int inputs, int depth, int hidden_layer_size, int outputs, double learning_speed = 0.5, double learning_add = 0.05,
		double momentum = 0.5, double momentum_sub = 0.01, double alpha = 1.7159, double beta = 2.0 / 3.0, double lambda = 0);
	neural_network(std::string file_name);
	void teach(std::vector<std::pair <std::vector<double>, std::vector<double> > >& tests,
		double error, int max_iterations);
	std::vector<double> calculate(std::vector<double> const& input);
	void save_to_file(std::string file_name);


private:
	void init();
	void forward_pass(std::vector<double> const& test);
	void backward_pass(std::vector<double> const& test_anwser);
	void normalize_old(std::vector<std::pair <std::vector<double>, std::vector<double> > > const& tests, double max_val, double min_freq);
	void normalize(std::vector<std::pair <std::vector<double>, std::vector<double> > > const& tests);
	void make_noise(std::default_random_engine& generator, std::normal_distribution<double>& distribution);

	std::vector<double> variance;
	std::vector<double> average;
	std::vector<layer> layers;
	int inputs;
	int outputs;
	int depth;
	int hidden_layer_size;
	double learning_speed;
	double learning_add;
	double momentum;
	double momentum_sub;
	double test_error;
	double alpha;
	double beta;
	double lambda;
	double tests_size;
};

#endif // NEURAL_NETWORK_H
