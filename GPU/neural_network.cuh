#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <string>
#include <random>
#include <curand_kernel.h>
#include <thrust\device_vector.h>

struct layer
{
	layer(int size, int inputs);
	layer();
	float* weights;
	float* delta_weights;
	float* outputs;
	float* borders;
	float* deltas;
	int inputs;
	int size;
};

class neural_network
{
public:
	neural_network(int inputs, int depth, int hidden_layer_size, int outputs, float learning_speed = 0.5, float learning_add = 0.05,
		float momentum = 0.5, float momentum_sub = 0.01, float alpha = 1.7159, float beta = 2.0 / 3.0, float lambda = 0);
	neural_network(std::string file_name);
	void teach(std::vector<std::pair <std::vector<float>, std::vector<float> > >& tests,
		float error, int max_iterations);
	std::vector<float> calculate(std::vector<float> const& input);
	void save_to_file(std::string file_name);

private:
	void init();
	void forward_pass(float* tests, int id, int size);
	void backward_pass(float* tests_anwsers, int id, int size, float* errors);
	void normalize_old(std::vector<std::pair <std::vector<float>, std::vector<float> > > const& tests, float max_val, float min_freq);
	void normalize(std::vector<std::pair <std::vector<float>, std::vector<float> > > const& tests);
	void make_noise(curandState* devStates);
	void make_noise_init(curandState* devStates);

	thrust::device_vector<float> variance;
	thrust::device_vector<float> average;
	thrust::device_vector<layer> layers;
	int inputs;
	int outputs;
	int depth;
	int hidden_layer_size;
	int max_dim;
	float learning_speed;
	float learning_add;
	float momentum;
	float momentum_sub;
	float alpha;
	float beta;
	float lambda;
	int tests_size;
};

#endif // NEURAL_NETWORK_H
