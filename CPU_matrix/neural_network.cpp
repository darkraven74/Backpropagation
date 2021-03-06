#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <random>
#include <cmath>

#include "neural_network.h"

void matrix_mul_cpu(double* a, double* b, double* c, int s1, int s2, int s3)
{
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
}

void matrix_mul_diagonal_cpu(double* a, double* b, double* c, int s1)
{
	for (int i = 0; i < s1; i++)
	{
		c[i] = a[i] * b[i];
	}
}

void matrix_mul_cpu(double* a, double b, double* c, int a_size)
{
	for (int i = 0; i < a_size; i++)
	{
		c[i] = a[i] * b;
	}
}

void matrix_add_cpu(double* a, double* b, double* c, int s1, int s2)
{
	for (int i = 0; i < s1; i++)
	{
		for (int j = 0; j < s2; j++)
		{
			c[i * s2 + j] = a[i * s2 + j] + b[i * s2 + j];
		}
	}
}

void matrix_func_cpu(double* a, double alpha, double beta, double a_size)
{
	for (int i = 0; i < a_size; i++)
	{
		a[i] = 1.0 * alpha * tanh(1.0 * beta * a[i]);
	}
}

void matrix_func_der_cpu(double* a, double* c, double alpha, double beta, double a_size)
{
	for (int i = 0; i < a_size; i++)
	{
		c[i] = 1.0 * beta / alpha * (alpha - a[i]) * (alpha + a[i]);
	}
}

void matrix_transpose_cpu(double* a, double* c, int s1, int s2)
{
	for (int i = 0; i < s1; i++)
	{
		for (int j = 0; j < s2; j++)
		{
			c[j * s1 + i] = a[i * s2 + j];
		}
	}
}

layer::layer(int size, int inputs) : size(size), inputs(inputs)
{
	weights = (double*)malloc(inputs * size * sizeof(double));
	delta_weights = (double*)malloc(inputs * size * sizeof(double));
	outputs = (double*)malloc(size * sizeof(double));
	deltas = (double*)malloc(size * sizeof(double));
	borders = (double*)malloc(size * sizeof(double));
	srand(time(NULL));
	//weights in min_w...max_w
	double min_w = inputs ? -1.0 / sqrt(1.0 * inputs) : -1.0;
	double max_w = inputs ? 1.0 / sqrt(1.0 * inputs) : 1.0;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < inputs; j++) 
		{
			weights[j * size + i] = ((max_w - min_w) * ((double)rand() / (double)RAND_MAX) + min_w);
		}
		borders[i] = ((max_w - min_w) * ((double)rand() / (double)RAND_MAX) + min_w);
	}
}

neural_network::neural_network(int inputs, int depth, int hidden_layer_size, int outputs, double learning_speed,
	double learning_add, double momentum, double momentum_sub, double alpha, double beta, double lambda)
	: inputs(inputs), depth(depth), hidden_layer_size(hidden_layer_size),
	  outputs(outputs), learning_speed(learning_speed), learning_add(learning_add), momentum(momentum),
	  momentum_sub(momentum_sub), alpha(alpha), beta(beta), lambda(lambda)
{
	init();
}

neural_network::neural_network(std::string file_name)
{
	FILE* f = fopen(file_name.c_str(), "r");
	fscanf(f, "%d %d %d %d", &inputs, &outputs, &depth, &hidden_layer_size);
	fscanf(f, "%lf %lf %lf %lf\n", &learning_speed, &momentum, &test_error, &alpha);
	variance.resize(inputs);
	for (int i = 0; i < variance.size(); i++)
	{
		fscanf(f, "%lf ", &variance[i]);
	}
	init();
	for (int i = 0; i < layers.size(); i++)
	{
		fscanf(f, "%d %d\n", &layers[i].inputs, &layers[i].size);
		for (int j = 0; j < layers[i].inputs * layers[i].size; j++)
		{
			fscanf(f, "%lf %lf ", &layers[i].weights[j], &layers[i].delta_weights[j]);
		}
		fscanf(f, "\n");
		for (int j = 0; j < layers[i].size; j++)
		{
			fscanf(f, "%lf %lf %lf ", &layers[i].outputs[j], &layers[i].borders[j], &layers[i].deltas[j]);
		}
	}
	fclose(f);
}

void neural_network::teach(std::vector<std::pair <std::vector<double>, std::vector<double> > >& tests, double error,
	int max_iterations)
{
	std::default_random_engine generator(time(NULL));
	std::normal_distribution<double> distribution(0.0, 1.0);
	
	tests_size = tests.size();
	//normalize_old(tests, 5.0, 1500.0);
	normalize(tests);
	clock_t time = clock();
	long long count = 0;
	double min_w = -0.2;
	double max_w = 0.2;
	double old_error;
	double curr_error = error + 10;
	while ((curr_error > error) && (count < max_iterations))
	{
		old_error = curr_error;
		count++;
		curr_error = 0;
		random_shuffle(tests.begin(), tests.end());
		//make_noise(generator, distribution);
		//std::vector<std::pair <std::vector<double>, std::vector<double> > > tests_noise = tests;
		/*for (int i = 0; i < tests_noise.size(); i++)
		{
			for (int j = 0; j < tests_noise[i].first.size(); j++)
			{
				tests_noise[i].first[j] += ((max_w - min_w) * ((double)rand() / (double)RAND_MAX) + min_w);
			}
		}*/
		for (int i = 0; i < tests.size(); i++)
		{
			//forward_pass(tests_noise[i].first);
			//backward_pass(tests_noise[i].second);
			forward_pass(tests[i].first);
			backward_pass(tests[i].second);
			curr_error += test_error;
		}
		curr_error /= tests.size();
		printf("ERROR: %f        count: %lld\n", curr_error, count);
		if (old_error == error + 10)
		{
			continue;
		}
		if (curr_error < old_error)
		{
			learning_speed *= 1.05;
		}
		if (curr_error - old_error > 1e-6)
		{
			learning_speed *= 0.5;
		}
		momentum -= momentum_sub;
	}
	printf("ERROR: %f\n", curr_error);
	printf("\ncount: %lld\n", count);
	time = clock() - time;
	printf("time: %f\n\n", (double)time / CLOCKS_PER_SEC);
}

std::vector<double> neural_network::calculate(std::vector<double> const& input)
{
	forward_pass(input);
	return std::vector<double> (layers[depth - 1].outputs, layers[depth - 1].outputs + layers[depth - 1].size);
}

void neural_network::save_to_file(std::string file_name)
{
	FILE* f = fopen(file_name.c_str(), "w");
	fprintf(f, "%d %d %d %d\n", inputs, outputs, depth, hidden_layer_size);
	fprintf(f, "%f %f %f %f\n", learning_speed, momentum, test_error, alpha);
	for (int i = 0; i < variance.size(); i++)
	{
		fprintf(f, "%f ", variance[i]);
	}
	fprintf(f, "\n");
	for (int i = 0; i < layers.size(); i++)
	{
		fprintf(f, "%d %d\n", layers[i].inputs, layers[i].size);
		for (int j = 0; j < layers[i].inputs * layers[i].size; j++)
		{
			fprintf(f, "%f %f ", layers[i].weights[j], layers[i].delta_weights[j]);
		}
		fprintf(f, "\n");
		for (int j = 0; j < layers[i].size; j++)
		{
			fprintf(f, "%f %f %f ", layers[i].outputs[j], layers[i].borders[j], layers[i].deltas[j]);
		}
		fprintf(f, "\n");
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
		layers[0].outputs[i] = test[i] - average[i];
		if (variance[i] > 1.0)
		{
			layers[0].outputs[i] /= (1.0 * variance[i]);
		}
	}
	/*for (int i = 0; i < inputs; i++)
	{
		layers[0].outputs[i] = 1.0 * test[i] / variance[i];
	}*/
	for (int i = 1; i < depth; i++)
	{
		matrix_mul_cpu(layers[i - 1].outputs, layers[i].weights, layers[i].outputs, 1, layers[i].inputs, layers[i].size);
		matrix_mul_cpu(layers[i].outputs, 1.0 / layers[i].inputs, layers[i].outputs, layers[i].size);
		matrix_add_cpu(layers[i].outputs, layers[i].borders, layers[i].outputs, 1, layers[i].size);
		matrix_func_cpu(layers[i].outputs, alpha, beta, layers[i].size);
	}
}

void neural_network::backward_pass(std::vector<double> const& test_anwser)
{
	test_error = 0;
	for (int i = 0; i < outputs; i++)
	{
		double curr_out = layers.back().outputs[i];
		test_error += (test_anwser[i] - curr_out) * (test_anwser[i] - curr_out);
		layers.back().deltas[i] = (test_anwser[i] - curr_out) * (alpha - curr_out) * (alpha + curr_out) * beta / alpha;  
		for (int j = 0; j < layers.back().inputs; j++)
		{
			layers.back().delta_weights[j * layers.back().size + i] = momentum * layers.back().delta_weights[j * layers.back().size + i]
			+ learning_speed * (layers.back().deltas[i] * layers[depth - 2].outputs[j] + lambda * 2.0 / tests_size * 
				layers.back().weights[j * layers.back().size + i]); 
		}
	}
	test_error /= 2;
	for (int i = depth - 2; i > 0; i--)
	{
		double local_learning_speed = learning_speed + (learning_add * (depth - i - 1));
		double* temp = (double*)malloc(layers[i + 1].size * layers[i + 1].inputs * sizeof(double));
		double* temp2 = (double*)malloc(layers[i + 1].inputs * sizeof(double));
		double* temp3 = (double*)malloc(layers[i].size * sizeof(double));
		matrix_transpose_cpu(layers[i + 1].weights, temp, layers[i + 1].inputs, layers[i + 1].size);
		matrix_mul_cpu(layers[i + 1].deltas, temp, temp2, 1, layers[i + 1].size, layers[i + 1].inputs);
		matrix_func_der_cpu(layers[i].outputs, temp3, alpha, beta, layers[i].size);
		matrix_mul_diagonal_cpu(temp3, temp2, layers[i].deltas, layers[i + 1].inputs);
		double* temp4 = (double *)malloc(layers[i].size * layers[i].inputs * sizeof(double));
		double* temp5 = (double *)malloc(layers[i].size * layers[i].inputs * sizeof(double));
		matrix_mul_cpu(layers[i].delta_weights, momentum, layers[i].delta_weights, layers[i].size * layers[i].inputs);
		matrix_mul_cpu(layers[i - 1].outputs, layers[i].deltas, temp4, layers[i].inputs, 1, layers[i].size);
		matrix_mul_cpu(layers[i].weights, 2.0 * lambda / tests_size, temp5, layers[i].size * layers[i].inputs);
		matrix_add_cpu(temp4, temp5, temp4, layers[i].inputs, layers[i].size);
		matrix_mul_cpu(temp4, local_learning_speed, temp4, layers[i].size * layers[i].inputs);
		matrix_add_cpu(layers[i].delta_weights, temp4, layers[i].delta_weights, layers[i].inputs, layers[i].size);
		free(temp);
		free(temp2);
		free(temp3);
		free(temp4);
		free(temp5);
	}
	for (int i = 1; i < depth; i++)
	{
		matrix_add_cpu(layers[i].weights, layers[i].delta_weights, layers[i].weights, layers[i].inputs, layers[i].size);
	}
}

void neural_network::normalize_old(std::vector<std::pair <std::vector<double>, std::vector<double> > > const& tests,
	double max_val, double min_freq)
{
	int n = tests[0].first.size();
	variance.resize(n, 1);
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
			variance[i] = 1.0 * sum[i] / freq[i];
		}
	}
}

void neural_network::normalize(std::vector<std::pair <std::vector<double>, std::vector<double> > > const& tests)
{
	int n = tests[0].first.size();
	variance.resize(n, 0);
	average.resize(n, 0);
	for (int i = 0; i < tests.size(); i++)
	{
		for (int j = 0; j < n; j++)
		{
			average[j] += tests[i].first[j];
		}
	}
	for (int i = 0; i < n; i++)
	{
		average[i] /= (1.0 * tests.size());
	}
	for (int i = 0; i < tests.size(); i++)
	{
		for (int j = 0; j < n; j++)
		{
			variance[j] += ((tests[i].first[j] - average[j]) * (tests[i].first[j] - average[j]));
		}
	}
	for (int i = 0; i < n; i++)
	{
		variance[i] /= (1.0 * tests.size());
		//variance[i] = sqrt(variance[i]);
	}
}

void neural_network::make_noise(std::default_random_engine& generator, std::normal_distribution<double>& distribution)
{
	for (int i = 0; i < depth; i++)
	{
		for (int j = 0; j < layers[i].size; j++)
		{
			layers[i].borders[j] = distribution(generator);
		}
	}
}