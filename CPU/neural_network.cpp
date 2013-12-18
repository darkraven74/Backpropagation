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

layer::layer(int size)
{

}


neural_network::neural_network(int inputs, int depth, int hidden_layer_size, int outputs)
	: inputs(inputs), depth(depth), hidden_layer_size(hidden_layer_size), outputs(outputs)
{

}