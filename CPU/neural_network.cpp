#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include "neural_network.h"

neuron::neuron(int inputs) : inputs(inputs)
{
	srand(time(NULL));
	//weights in min_w...max_w
	double min_w = -0.2;
	double max_w = 0.2;
	for (int i = 0; i < inputs; i++) 
	{
		weights.push_back((max_w - min_w) * ((double)rand() / (double)RAND_MAX) + min_w);
	}
	border = (max_w - min_w) * ((double)rand() / (double)RAND_MAX) + min_w;
	//border = 0;
	delta_weights.resize(inputs);
}

layer::layer(int size, int inputs) : size(size)
{
	for (int i = 0; i < size; i++)
	{
		neurons.push_back(neuron(inputs));
	}
}

neural_network::neural_network(int inputs, int depth, int hidden_layer_size, int outputs, double learning_speed,
	double momentum, double alpha)
	: inputs(inputs), depth(depth), hidden_layer_size(hidden_layer_size),
	  outputs(outputs), learning_speed(learning_speed), momentum(momentum), alpha(alpha)
{
	init();
}

neural_network::neural_network(std::string file_name)
{
	FILE* f = fopen(file_name.c_str(), "r");
	fscanf(f, "%d %d %d %d", &inputs, &outputs, &depth, &hidden_layer_size);
	fscanf(f, "%lf %lf %lf %lf\n", &learning_speed, &momentum, &test_error, &alpha);
	coeff.resize(inputs);
	for (int i = 0; i < coeff.size(); i++)
	{
		fscanf(f, "%lf ", &coeff[i]);
	}
	init();
	for (int i = 0; i < layers.size(); i++)
	{
		fscanf(f, "%d", &layers[i].size);
		for (int j = 0; j < layers[i].neurons.size(); j++)
		{
			fscanf(f, "%lf %lf %lf %d", &layers[i].neurons[j].output, &layers[i].neurons[j].border, &layers[i].neurons[j].delta,
				&layers[i].neurons[j].inputs);
			for (int k = 0; k < layers[i].neurons[j].weights.size(); k++)
			{
				fscanf(f, "%lf ", &layers[i].neurons[j].weights[k]);
			}
			for (int k = 0; k < layers[i].neurons[j].delta_weights.size(); k++)
			{
				fscanf(f, "%lf ", &layers[i].neurons[j].delta_weights[k]);
			}
		}
	}
	fclose(f);
}

void neural_network::teach(std::vector<std::pair <std::vector<double>, std::vector<double> > >& tests, double error,
	int max_iterations, double max_val, double min_freq)
{
	normalize(tests, max_val, min_freq);
	clock_t time = clock();
	long long count = 0;
	double curr_error = error + 1;
	while ((curr_error > error) && (count < max_iterations))
	{
		count++;
		curr_error = 0;
		random_shuffle(tests.begin(), tests.end());
		for (int i = 0; i < tests.size(); i++)
		{
			forward_pass(tests[i].first);
			backward_pass(tests[i].second);
			curr_error += test_error;
		}
		curr_error /= tests.size();
		printf("ERROR: %f        count: %lld\n", curr_error, count);
	}
	printf("ERROR: %f\n", curr_error);
	printf("\ncount: %lld\n", count);
	time = clock() - time;
	printf("time: %f\n\n", (double)time / CLOCKS_PER_SEC);
}

std::vector<double> neural_network::calculate(std::vector<double> const& input)
{
	std::vector<double> anwser(outputs);
	forward_pass(input);
	for (int i = 0; i < outputs; i++)
	{
		anwser[i] = layers.back().neurons[i].output;
	}
	return anwser;
}

void neural_network::save_to_file(std::string file_name)
{
	FILE* f = fopen(file_name.c_str(), "w");
	fprintf(f, "%d %d %d %d\n", inputs, outputs, depth, hidden_layer_size);
	fprintf(f, "%f %f %f %f\n", learning_speed, momentum, test_error, alpha);
	for (int i = 0; i < coeff.size(); i++)
	{
		fprintf(f, "%f ", coeff[i]);
	}
	fprintf(f, "\n");
	for (int i = 0; i < layers.size(); i++)
	{
		fprintf(f, "%d\n", layers[i].size);
		for (int j = 0; j < layers[i].neurons.size(); j++)
		{
			fprintf(f, "%f %f %f %d\n", layers[i].neurons[j].output, layers[i].neurons[j].border, layers[i].neurons[j].delta,
				 layers[i].neurons[j].inputs);
			for (int k = 0; k < layers[i].neurons[j].weights.size(); k++)
			{
				fprintf(f, "%f ", layers[i].neurons[j].weights[k]);
			}
			fprintf(f, "\n");
			for (int k = 0; k < layers[i].neurons[j].delta_weights.size(); k++)
			{
				fprintf(f, "%f ", layers[i].neurons[j].delta_weights[k]);
			}
			fprintf(f, "\n");
		}
	}
	fclose(f);
}

void neural_network::init()
{
	layers.push_back(layer(inputs, 0));
	for (int i = 1; i < depth - 1; i++)
	{
		layers.push_back(layer(hidden_layer_size, layers[i - 1].size));
	}
	layers.push_back(layer(outputs, hidden_layer_size));
}

void neural_network::forward_pass(std::vector<double> const& test)
{
	for (int i = 0; i < inputs; i++)
	{
		layers[0].neurons[i].output = test[i] / coeff[i];
	}
	for (int i = 1; i < depth; i++)
	{
		for (int j = 0; j < layers[i].size; j++)
		{
			double arg = layers[i].neurons[j].border;
			for (int k = 0; k < layers[i].neurons[j].inputs; k++)
			{
				arg += (layers[i - 1].neurons[k].output * layers[i].neurons[j].weights[k]);
			}
			layers[i].neurons[j].output = 1.0 / (1.0 + exp(-1.0 * arg * alpha));
		}
	}
}

void neural_network::backward_pass(std::vector<double> const& test_anwser)
{
	test_error = 0;
	for (int i = 0; i < outputs; i++)
	{
		double curr_out = layers.back().neurons[i].output;
		test_error += (test_anwser[i] - curr_out) * (test_anwser[i] - curr_out);
		layers.back().neurons[i].delta = (test_anwser[i] - curr_out) * curr_out * (1.0 - curr_out) * alpha;  
		for (int j = 0; j < layers.back().neurons[i].inputs; j++)
		{
			layers.back().neurons[i].delta_weights[j] = momentum * layers.back().neurons[i].delta_weights[j]
			 + learning_speed * layers.back().neurons[i].delta * layers[depth - 2].neurons[j].output; 
		}
	}
	test_error /= 2;
	for (int i = depth - 2; i >= 0; i--)
	{
		for (int j = 0; j < layers[i].size; j++)
		{
			double sum = 0;
			for (int k = 0; k < layers[i + 1].size; k++)
			{
				sum += layers[i + 1].neurons[k].delta * layers[i + 1].neurons[k].weights[j];
			}
			layers[i].neurons[j].delta = sum * layers[i].neurons[j].output * (1.0 - layers[i].neurons[j].output) * alpha;
			for (int k = 0; k < layers[i].neurons[j].inputs; k++)
			{
				layers[i].neurons[j].delta_weights[k] = momentum * layers[i].neurons[j].delta_weights[k]
				+ learning_speed * layers[i].neurons[j].delta * layers[i - 1].neurons[k].output; 
			}
		}
	}
	for (int i = 0; i < depth; i++)
	{
		for (int j = 0; j < layers[i].size; j++)
		{
			for (int k = 0; k < layers[i].neurons[j].inputs; k++)
			{
				layers[i].neurons[j].weights[k] += layers[i].neurons[j].delta_weights[k]; 
			}
		}
	}
}

void neural_network::normalize(std::vector<std::pair <std::vector<double>, std::vector<double> > > const& tests,
	double max_val, double min_freq)
{
	int n = tests[0].first.size();
	coeff.resize(n, 1);
	std::vector<double> sum(n);
	std::vector<int> freq(n);
	for (int i = 0; i < tests.size(); i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (tests[i].first[j] > max_val || tests[i].first[j] < -max_val)
			{
				freq[j]++;
				sum[j] += tests[i].first[j];
			}
		}
	}
	for (int i = 0; i < n; i++)
	{
		if (freq[i] > min_freq)
		{
			coeff[i] = 1.0 * sum[i] / freq[i];
		}
	}
}