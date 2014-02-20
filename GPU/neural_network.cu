#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <cmath>
#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include "neural_network.cuh"

#define BLOCK_SIZE 32

__device__ void matrix_mul_gpu(float* a, float* b, float* c, int s1, int s2, int s3)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < s1 && j < s3)
	{
		float sum = 0;
		for (int k = 0; k < s2; k++)
		{
			sum += a[i * s2 + k] * b[k * s3 + j];
		}
		c[i * s3 + j] = sum;
	}
}

__device__ void matrix_mul_gpu(float* a, float b, float* c, int s1, int s2)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j < s2 && i < s1)
	{
		c[i * s2 + j] = a[i * s2 + j] * b;
	}
}

__device__ void matrix_mul_diagonal_gpu(float* a, float* b, float* c, int s1)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j == 0 && i < s1)
	{
		c[i] = a[i] * b[i];
	}
}

__device__ void matrix_add_gpu(float* a, float* b, float* c, int s1, int s2)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < s1 && j < s2)
	{
		c[i * s2 + j] = a[i * s2 + j] + b[i * s2 + j];
	}
}

__device__ void matrix_transpose_gpu(float* a, float* c, int s1, int s2)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < s1 && j < s2)
	{
		c[j * s1 + i] = a[i * s2 + j];
	}
}

__device__ void matrix_func_gpu(float* a, float alpha, float a_size)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i == 0 && j < a_size)
	{
		a[j] = 1.0f / (1.0f + __expf(-1.0f * a[j] * alpha));
	}
}

__device__ void matrix_func_der_gpu(float* a, float* c, float alpha, float a_size)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j == 0 && i < a_size)
	{
		c[i] = a[i] * alpha * (1.0f - a[i]);
	}
}

layer::layer(int size, int inputs) : size(size), inputs(inputs)
{
	float* h_weights = (float*)malloc(inputs * size * sizeof(float));
	float* h_borders = (float*)malloc(size * sizeof(float));
	cudaMalloc(&weights, inputs * size * sizeof(float));
	cudaMalloc(&delta_weights, inputs * size * sizeof(float));
	cudaMalloc(&outputs, size * sizeof(float));
	cudaMalloc(&deltas, size * sizeof(float));
	cudaMalloc(&borders, size * sizeof(float));
	srand(time(NULL));
	//weights in min_w...max_w
	float min_w = -0.2f;
	float max_w = 0.2f;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < inputs; j++) 
		{
			h_weights[j * size + i] = ((max_w - min_w) * ((float)rand() / (float)RAND_MAX) + min_w);
		}
		h_borders[i] = ((max_w - min_w) * ((float)rand() / (float)RAND_MAX) + min_w);
	}
	cudaMemcpy(weights, h_weights, inputs * size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(borders, h_borders, size * sizeof(float), cudaMemcpyHostToDevice);
	free(h_weights);
	free(h_borders);
}

layer::layer()
{

}

neural_network::neural_network(int inputs, int depth, int hidden_layer_size, int outputs, float learning_speed,
	float momentum, float alpha)
	: inputs(inputs), depth(depth), hidden_layer_size(hidden_layer_size),
	  outputs(outputs), learning_speed(learning_speed), momentum(momentum), alpha(alpha)
{
	init();
}

neural_network::neural_network(std::string file_name)
{
	FILE* f = fopen(file_name.c_str(), "r");
	fscanf(f, "%d %d %d %d", &inputs, &outputs, &depth, &hidden_layer_size);
	fscanf(f, "%f %f %f\n", &learning_speed, &momentum, &alpha);
	thrust::host_vector<float> coeff_h(inputs);
	for (int i = 0; i < inputs; i++)
	{
		fscanf(f, "%f ", &coeff_h[i]);
	}
	coeff = coeff_h;
	init();
	for (int i = 0; i < depth; i++)
	{
		layer l = layers[i];
		int x, y;
		fscanf(f, "%d %d\n", &x, &y);
		float* weights_h = (float*)malloc(l.inputs * l.size * sizeof(float));
		float* delta_weights_h = (float*)malloc(l.inputs * l.size * sizeof(float));
		float* outputs_h = (float*)malloc(l.size * sizeof(float));
		float* borders_h = (float*)malloc(l.size * sizeof(float));
		float* deltas_h = (float*)malloc(l.size * sizeof(float));
		for (int j = 0; j < l.inputs * l.size; j++)
		{
			fscanf(f, "%f %f ", &weights_h[j], &delta_weights_h[j]);
		}
		fscanf(f, "\n");
		for (int j = 0; j < l.size; j++)
		{
			fscanf(f, "%f %f %f ", &outputs_h[j], &borders_h[j], &deltas_h[j]);
		}
		cudaMemcpy(l.weights, weights_h, l.inputs * l.size * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(l.delta_weights, delta_weights_h, l.inputs * l.size * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(l.outputs, outputs_h, l.size * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(l.borders, borders_h, l.size * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(l.deltas, deltas_h, l.size * sizeof(float), cudaMemcpyHostToDevice);
		free(weights_h);
		free(delta_weights_h);
		free(outputs_h);
		free(borders_h);
		free(deltas_h);
	}
	fclose(f);
}


void neural_network::teach(std::vector<std::pair <std::vector<float>, std::vector<float> > >& tests, float error,
	int max_iterations, float max_val, float min_freq)
{
	normalize(tests, max_val, min_freq);
	clock_t time = clock();
	long long count = 0;
	float curr_error = error + 1;
	while ((curr_error > error) && (count < max_iterations))
	{
		count++;
		curr_error = 0;
		random_shuffle(tests.begin(), tests.end());
		float* tests_h = (float*)malloc(tests.size() * tests[0].first.size() * sizeof(float));
		float* tests_anwsers_h = (float*)malloc(tests.size() * tests[0].second.size() * sizeof(float));
		for (int i = 0; i < tests.size(); i++)
		{
			copy(tests[i].first.begin(), tests[i].first.end(), tests_h + i * tests[0].first.size());
			copy(tests[i].second.begin(), tests[i].second.end(), tests_anwsers_h + i * tests[0].second.size());
		}
		float* tests_d;
		float* tests_anwsers_d;
		cudaMalloc(&tests_d, tests.size() * tests[0].first.size() * sizeof(float));
		cudaMalloc(&tests_anwsers_d, tests.size() * tests[0].second.size() * sizeof(float));
		cudaMemcpy(tests_d, tests_h, tests.size() * tests[0].first.size() * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(tests_anwsers_d, tests_anwsers_h, tests.size() * tests[0].second.size() * sizeof(float), cudaMemcpyHostToDevice);
		thrust::device_vector<float> errors_d(tests.size());
		for (int i = 0; i < tests.size(); i++)
		{
			forward_pass(tests_d, i, tests[0].first.size());
			backward_pass(tests_anwsers_d, i, tests[0].second.size(), thrust::raw_pointer_cast(&errors_d[0]));
		}
		cudaDeviceSynchronize();
		thrust::host_vector<float> errors_h = errors_d;
		cudaDeviceSynchronize();
		for (int i = 0; i < tests.size(); i++)
		{
			curr_error += errors_h[i];
		}
		free(tests_h);
		free(tests_anwsers_h);
		cudaFree(tests_d);
		cudaFree(tests_anwsers_d);
		curr_error /= tests.size();
		printf("ERROR: %f        count: %lld\n", curr_error, count);
	}
	printf("ERROR: %f\n", curr_error);
	printf("\ncount: %lld\n", count);
	time = clock() - time;
	printf("time: %f\n\n", (float)time / CLOCKS_PER_SEC);
}

std::vector<float> neural_network::calculate(std::vector<float> const& input)
{
	float* input_d;
	float* ans_h = (float*)malloc(outputs * sizeof(float));
	cudaMalloc(&input_d, input.size() * sizeof(float));
	cudaMemcpy(input_d, &input[0], input.size() * sizeof(float), cudaMemcpyHostToDevice);
	forward_pass(input_d, 0, input.size());
	layer l = layers[depth - 1];
	cudaMemcpy(ans_h, l.outputs, outputs * sizeof(float), cudaMemcpyDeviceToHost);
	std::vector<float> ans(ans_h, ans_h + outputs);
	cudaFree(input_d);
	free(ans_h);
	return ans;
}

void neural_network::save_to_file(std::string file_name)
{
	FILE* f = fopen(file_name.c_str(), "w");
	fprintf(f, "%d %d %d %d\n", inputs, outputs, depth, hidden_layer_size);
	fprintf(f, "%f %f %f\n", learning_speed, momentum, alpha);
	thrust::host_vector<float> coeff_h = coeff;
	for (int i = 0; i < coeff_h.size(); i++)
	{
		fprintf(f, "%f ", coeff_h[i]);
	}
	fprintf(f, "\n");
	for (int i = 0; i < depth; i++)
	{
		layer l = layers[i];
		fprintf(f, "%d %d\n", l.inputs, l.size);
		float* weights_h = (float*)malloc(l.inputs * l.size * sizeof(float));
		float* delta_weights_h = (float*)malloc(l.inputs * l.size * sizeof(float));
		float* outputs_h = (float*)malloc(l.size * sizeof(float));
		float* borders_h = (float*)malloc(l.size * sizeof(float));
		float* deltas_h = (float*)malloc(l.size * sizeof(float));
		cudaMemcpy(weights_h, l.weights, l.inputs * l.size * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(delta_weights_h, l.delta_weights, l.inputs * l.size * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(outputs_h, l.outputs, l.size * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(borders_h, l.borders, l.size * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(deltas_h, l.deltas, l.size * sizeof(float), cudaMemcpyDeviceToHost);
		for (int j = 0; j < l.inputs * l.size; j++)
		{
			fprintf(f, "%f %f ", weights_h[j], delta_weights_h[j]);
		}
		fprintf(f, "\n");
		for (int j = 0; j < l.size; j++)
		{
			fprintf(f, "%f %f %f ", outputs_h[j], borders_h[j], deltas_h[j]);
		}
		fprintf(f, "\n");
		free(weights_h);
		free(delta_weights_h);
		free(outputs_h);
		free(borders_h);
		free(deltas_h);
	}
	fclose(f);
}

void neural_network::init()
{
	max_dim = std::max(inputs, std::max(outputs, hidden_layer_size));
	layers.push_back(layer(inputs, 0));
	layers.push_back(layer(hidden_layer_size, inputs));
	for (int i = 2; i < depth - 1; i++)
	{
		layers.push_back(layer(hidden_layer_size, hidden_layer_size));
	}
	layers.push_back(layer(outputs, hidden_layer_size));
}

__global__ void help_forward_pass_gpu(layer* layers, float* test, float* coeff, int id, int size)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	layers[0].outputs[j] = test[id * size + j] / coeff[j];
}

__global__ void forward_pass_gpu(layer* layers, int i, float alpha)
{
	matrix_mul_gpu(layers[i - 1].outputs, layers[i].weights, layers[i].outputs, 1, layers[i].inputs, layers[i].size);
	__syncthreads();
	matrix_add_gpu(layers[i].outputs, layers[i].borders, layers[i].outputs, 1, layers[i].size);
	__syncthreads();
	matrix_func_gpu(layers[i].outputs, alpha, layers[i].size);
}

void neural_network::forward_pass(float* tests, int id, int size)
{
	dim3 block(BLOCK_SIZE, 1);
	dim3 grid(1 + size / (1 + BLOCK_SIZE), 1);
	help_forward_pass_gpu<<<grid, block>>>(thrust::raw_pointer_cast(&layers[0]), tests,
		thrust::raw_pointer_cast(&coeff[0]), id, size);
	cudaDeviceSynchronize();
	block.y = BLOCK_SIZE;
	grid.x = 1 + max_dim / (1 + BLOCK_SIZE);
	for (int i = 1; i < depth; i++)
	{
		forward_pass_gpu<<<grid, block>>>(thrust::raw_pointer_cast(&layers[0]), i, alpha);
		cudaDeviceSynchronize();
	}
}

__global__ void help_backward_pass_gpu(layer* layers, float* tests_anwsers, int depth, float alpha, float momentum,
	 float learning_speed, float* errors, int id, int size)
{
	errors[id] = 0;
	for (int i = 0; i < size; i++)
	{
		float curr_out = layers[depth - 1].outputs[i];
		errors[id] += (tests_anwsers[id * size + i] - curr_out) * (tests_anwsers[id * size + i] - curr_out);
		layers[depth - 1].deltas[i] = (tests_anwsers[id * size + i] - curr_out) * curr_out * (1.0 - curr_out) * alpha;  
		for (int j = 0; j < layers[depth - 1].inputs; j++)
		{
			layers[depth - 1].delta_weights[j * layers[depth - 1].size + i] = momentum *
				layers[depth - 1].delta_weights[j * layers[depth - 1].size + i] + learning_speed * layers[depth - 1].deltas[i] *
				layers[depth - 2].outputs[j]; 
		}
	}
	errors[id] /= 2;
}

__global__ void backward_pass_gpu(layer* layers, float* temp, float* temp2, float* temp3, float* temp4,
	 float alpha, float momentum, float learning_speed, int i)
{
	matrix_transpose_gpu(layers[i + 1].weights, temp, layers[i + 1].inputs, layers[i + 1].size);
	__syncthreads();
	matrix_mul_gpu(layers[i + 1].deltas, temp, temp2, 1, layers[i + 1].size, layers[i + 1].inputs);
	__syncthreads();
	matrix_func_der_gpu(layers[i].outputs, temp3, alpha, layers[i].size);
	__syncthreads();
	matrix_mul_diagonal_gpu(temp3, temp2, layers[i].deltas, layers[i + 1].inputs);
	__syncthreads();
	matrix_mul_gpu(layers[i].delta_weights, momentum, layers[i].delta_weights, layers[i].inputs, layers[i].size);
	__syncthreads();
	matrix_mul_gpu(layers[i - 1].outputs, layers[i].deltas, temp4, layers[i].inputs, 1, layers[i].size);
	__syncthreads();
	matrix_mul_gpu(temp4, learning_speed, temp4, layers[i].inputs, layers[i].size);
	__syncthreads();
	matrix_add_gpu(layers[i].delta_weights, temp4, layers[i].delta_weights, layers[i].inputs, layers[i].size);
}

__global__ void help2_backward_pass_gpu(layer* layers, int i)
{
	matrix_add_gpu(layers[i].weights, layers[i].delta_weights, layers[i].weights, layers[i].inputs, layers[i].size);
}

void neural_network::backward_pass(float* tests_anwsers, int id, int size, float* errors)
{
	dim3 block(1, 1);
	dim3 grid(1, 1);
	help_backward_pass_gpu<<<grid, block>>>(thrust::raw_pointer_cast(&layers[0]), tests_anwsers, depth, alpha,
		momentum, learning_speed, errors, id, size);
	cudaDeviceSynchronize(); 
	block.x = BLOCK_SIZE;
	block.y = BLOCK_SIZE;
	grid.x = 1 + max_dim / (1 + BLOCK_SIZE);
	grid.y = 1 + max_dim / (1 + BLOCK_SIZE);
	for (int i = depth - 2; i > 0; i--)
	{
		float* temp;
		float* temp2;
		float* temp3;
		float* temp4;
		cudaMalloc(&temp, outputs * hidden_layer_size * sizeof(float));
		cudaMalloc(&temp2, hidden_layer_size * sizeof(float));
		cudaMalloc(&temp3, hidden_layer_size * sizeof(float));
		cudaMalloc(&temp4, hidden_layer_size * inputs * sizeof(float));
		backward_pass_gpu<<<grid, block>>>(thrust::raw_pointer_cast(&layers[0]),
			temp, temp2, temp3, temp4, alpha, momentum, learning_speed, i);
		cudaDeviceSynchronize();
		cudaFree(temp);
		cudaFree(temp2);
		cudaFree(temp3);
		cudaFree(temp4);
	}
	for (int i = 1; i < depth; i++)
	{
		help2_backward_pass_gpu<<<grid, block>>>(thrust::raw_pointer_cast(&layers[0]), i);
		cudaDeviceSynchronize();
	}
}

void neural_network::normalize(std::vector<std::pair <std::vector<float>, std::vector<float> > > const& tests,
	float max_val, float min_freq)
{
	int n = tests[0].first.size();
	coeff.resize(n, 1);
	std::vector<float> sum(n);
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
			coeff[i] = 1.0f * sum[i] / freq[i];
		}
	}
}