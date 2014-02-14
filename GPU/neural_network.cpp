#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <cmath>



#include "neural_network.h"

std::vector<double> matrix_mul_cpu(std::vector<double> const& a, std::vector<double> const& b, int s1, int s3)
{
	int s2 = a.size() / s1;
	std::vector<double> c(s1 * s3);
	for (int i = 0; i < s1; i++)
	{
		for (int j = 0; j < s3; j++)
		{
			double sum = 0;
			for (int k = 0; k < s2; k++)
			{
				sum += a[i * s2 + k] * b[k * s3 + j];
			}
			c[i * s3 + j] = sum;
		}
	}
	return c;
}

std::vector<double> matrix_mul_diagonal_cpu(std::vector<double> const& a, std::vector<double> const& b, int s1)
{
	std::vector<double> c(s1);
	for (int i = 0; i < s1; i++)
	{
		c[i] = a[i] * b[i];
	}
	return c;
}

std::vector<double> matrix_mul_cpu(std::vector<double> const& a, double b)
{
	std::vector<double> c(a.size());
	for (int i = 0; i < a.size(); i++)
	{
		c[i] = a[i] * b;
	}
	return c;
}

std::vector<double> matrix_add_cpu(std::vector<double> const& a, std::vector<double> const& b, int s1)
{
	int s2 = a.size() / s1;
	std::vector<double> c(a.size());
	for (int i = 0; i < s1; i++)
	{
		for (int j = 0; j < s2; j++)
		{
			c[i * s2 + j] = a[i * s2 + j] + b[i * s2 + j];
		}
	}
	return c;
}

void matrix_func_cpu(std::vector<double>& a, double alpha)
{
	for (int i = 0; i < a.size(); i++)
	{
		a[i] = 1.0 / (1.0 + exp(-1.0 * a[i] * alpha));
	}
}

std::vector<double> matrix_func_der_cpu(std::vector<double> const& a, double alpha)
{
	std::vector<double> c(a.size());
	for (int i = 0; i < a.size(); i++)
	{
		c[i] = a[i] * alpha * (1.0 - a[i]);
	}
	return c;
}

std::vector<double> matrix_transpose_cpu(std::vector<double>& a, int s1)
{
	std::vector<double> c(a.size());
	int s2 = a.size() / s1;
	for (int i = 0; i < s1; i++)
	{
		for (int j = 0; j < s2; j++)
		{
			c[j * s1 + i] = a[i * s2 + j];
		}
	}
	return c;
}

layer::layer(int size, int inputs) : size(size), inputs(inputs)
{
	weights.resize(inputs * size);
	delta_weights.resize(inputs * size);
	outputs.resize(size);
	deltas.resize(size);
	srand(time(NULL));
	//weights in min_w...max_w
	double min_w = -0.2;
	double max_w = 0.2;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < inputs; j++) 
		{
			weights[j * size + i] = ((max_w - min_w) * ((double)rand() / (double)RAND_MAX) + min_w);
		}
		borders.push_back((max_w - min_w) * ((double)rand() / (double)RAND_MAX) + min_w);
		//borders.push_back(0);
	}
}

neural_network::neural_network(int inputs, int depth, int hidden_layer_size, int outputs, double learning_speed,
	double momentum, double alpha)
	: inputs(inputs), depth(depth), hidden_layer_size(hidden_layer_size),
	  outputs(outputs), learning_speed(learning_speed), momentum(momentum), alpha(alpha)
{
	init();
}

/*neural_network::neural_network(std::string file_name)
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
}*/

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
	forward_pass(input);
	return layers.back().outputs;
}

/*void neural_network::save_to_file(std::string file_name)
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
}*/

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
		layers[0].outputs[i] = test[i] / coeff[i];
	}
	for (int i = 1; i < depth; i++)
	{
		layers[i].outputs = matrix_add_cpu(matrix_mul_cpu(layers[i - 1].outputs, layers[i].weights, 1, layers[i].size),
			layers[i].borders, layers[i].size); 
		matrix_func_cpu(layers[i].outputs, alpha);
	}
}

void neural_network::backward_pass(std::vector<double> const& test_anwser)
{
	test_error = 0;
	for (int i = 0; i < outputs; i++)
	{
		double curr_out = layers.back().outputs[i];
		test_error += (test_anwser[i] - curr_out) * (test_anwser[i] - curr_out);
		layers.back().deltas[i] = (test_anwser[i] - curr_out) * curr_out * (1.0 - curr_out) * alpha;  
		for (int j = 0; j < layers.back().inputs; j++)
		{
			layers.back().delta_weights[j * layers.back().size + i] = momentum * layers.back().delta_weights[j * layers.back().size + i]
			+ learning_speed * layers.back().deltas[i] * layers[depth - 2].outputs[j]; 
		}
	}
	test_error /= 2;
	for (int i = depth - 2; i > 0; i--)
	{
		layers[i].deltas = matrix_mul_diagonal_cpu(matrix_func_der_cpu(layers[i].outputs, alpha),
			 matrix_mul_cpu(layers[i + 1].deltas, matrix_transpose_cpu(layers[i + 1].weights, layers[i + 1].inputs),
			  1, layers[i + 1].inputs), layers[i + 1].inputs);
		layers[i].delta_weights = matrix_add_cpu(matrix_mul_cpu(layers[i].delta_weights, momentum),
			matrix_mul_cpu(matrix_mul_cpu(layers[i - 1].outputs, layers[i].deltas, layers[i].inputs, layers[i].size), learning_speed),
			layers[i].inputs);
	}
	for (int i = 1; i < depth; i++)
	{
		layers[i].weights = matrix_add_cpu(layers[i].weights, layers[i].delta_weights, layers[i].inputs);
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
			if (abs(tests[i].first[j]) > max_val)
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