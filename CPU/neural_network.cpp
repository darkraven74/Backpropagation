#include <stdlib.h>
#include <time.h>   
#include <algorithm>
#include <math.h>
#include "neural_network.h"


neuron::neuron(int inputs) : inputs(inputs)
{
	srand(time(NULL));
	border = (double)rand() / (double)RAND_MAX;
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

void neural_network::teach(vector<pair <vector<double>, vector<double> > > tests)
{
	random_shuffle(tests.begin(), tests.end());
	for (int i = 0; i < tests.size(); i++)
	{
		forward_pass(tests[i].first);
		//...
	}

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

void neural_network::backward_pass()
{

}