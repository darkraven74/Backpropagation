#include <stdlib.h>
#include <time.h>   
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include "neural_network.h"

neuron::neuron(int inputs) : inputs(inputs)
{
	srand(time(NULL));
	//border = (double)rand() / (double)RAND_MAX;
	border = 0;
	for (int i = 0; i < inputs; i++) 
	{
		weights.push_back((double)rand() / (double)RAND_MAX);
	}
	delta_weights.resize(inputs);
}

layer::layer(int size, int inputs) : size(size)
{
	for (int i = 0; i < size; i++)
	{
		neurons.push_back(neuron(inputs));
	}
}


neural_network::neural_network(int inputs, int depth, int hidden_layer_size, int outputs)
	: inputs(inputs), depth(depth), hidden_layer_size(hidden_layer_size),
	  outputs(outputs), learning_speed(0.5), momentum(0.5)	
{
	init();
}

neural_network::neural_network(int inputs, int depth, int hidden_layer_size, int outputs, double learning_speed, double momentum)
	: inputs(inputs), depth(depth), hidden_layer_size(hidden_layer_size),
	  outputs(outputs), learning_speed(learning_speed), momentum(momentum)
{
	init();
}

void neural_network::teach(vector<pair <vector<double>, vector<double> > > tests, double error)
{
	clock_t time = clock();
	long long count = 0;
	double curr_error = error + 1;
	while (curr_error > error)
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
		printf("ERROR: %f\n", curr_error);
	}
	printf("\ncount: %lld\n", count);
	time = clock() - time;
	printf("time: %f\n\n", (double)time / CLOCKS_PER_SEC);
}

vector<double> neural_network::calculate(vector<double> input)
{
	vector<double> anwser(outputs);
	forward_pass(input);
	for (int i = 0; i < outputs; i++)
	{
		anwser[i] = layers.back().neurons[i].output;
	}
	return anwser;
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

void neural_network::forward_pass(vector<double>& test)
{
	for (int i = 0; i < inputs; i++)
	{
		layers[0].neurons[i].output = test[i];
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
			layers[i].neurons[j].output = 1.0 / (1.0 + exp(-1.0 * arg));
		}
	}
}

void neural_network::backward_pass(vector<double>& test_anwser)
{
	test_error = 0;
	for (int i = 0; i < outputs; i++)
	{
		double curr_out = layers.back().neurons[i].output;
		test_error += (test_anwser[i] - curr_out) * (test_anwser[i] - curr_out);
		layers.back().neurons[i].delta = (test_anwser[i] - curr_out) * curr_out * (1.0 - curr_out);  
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
			layers[i].neurons[j].delta = sum * layers[i].neurons[j].output * (1.0 - layers[i].neurons[j].output);
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