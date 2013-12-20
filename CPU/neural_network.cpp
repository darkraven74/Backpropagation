#include <stdlib.h>
#include <time.h>   
#include "neural_network.h"


neuron::neuron(int inputs)
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
	: inputs(inputs), depth(depth), hidden_layer_size(hidden_layer_size), outputs(outputs)
{
	layers.push_back(layer(inputs, 0));
	for (int i = 1; i < depth - 1; i++)
	{
		layers.push_back(layer(hidden_layer_size, layers[i - 1].size));
	}
	layers.push_back(layer(outputs, hidden_layer_size));
}