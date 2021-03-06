#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <cmath>
#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include <curand_kernel.h>
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

__device__ void matrix_func_gpu(float* a, float alpha, float beta, float a_size)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i == 0 && j < a_size)
	{
		a[j] = 1.0f * alpha * tanh(1.0 * beta * a[j]);
	}
}

__device__ void matrix_func_der_gpu(float* a, float* c, float alpha, float beta, float a_size)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j == 0 && i < a_size)
	{
		c[i] = 1.0 * beta / alpha * (alpha - a[i]) * (alpha + a[i]);
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
	float min_w = inputs ? -1.0 / sqrt(1.0 * inputs) : -1.0;
 	float max_w = inputs ? 1.0 / sqrt(1.0 * inputs) : 1.0;
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
	float learning_add, float momentum, float momentum_sub, float alpha, float beta, float lambda)
	: outputs(outputs), learning_speed(learning_speed), learning_add(learning_add), momentum(momentum),
	  momentum_sub(momentum_sub), alpha(alpha), beta(beta), lambda(lambda)
{
	init();
}

neural_network::neural_network(std::string file_name)
{
	FILE* f = fopen(file_name.c_str(), "r");
	fscanf(f, "%d %d %d %d", &inputs, &outputs, &depth, &hidden_layer_size);
	fscanf(f, "%f %f %f\n", &learning_speed, &momentum, &alpha);
	thrust::host_vector<float> variance_h(inputs);
	for (int i = 0; i < inputs; i++)
	{
		fscanf(f, "%f ", &variance_h[i]);
	}
	variance = variance_h;
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
	int max_iterations)
{
	tests_size = tests.size();
	normalize(tests);
	clock_t time = clock();
	curandState* devStates;
    cudaMalloc(&devStates, max_dim * sizeof(curandState));
	make_noise_init(devStates);
	long long count = 0;
	float min_w = -0.2;
	float max_w = 0.2;
	float curr_error = error + 10;
	float old_error;
	while ((curr_error > error) && (count < max_iterations))
	{
		old_error = curr_error;
		count++;
		curr_error = 0;
		//make_noise(devStates);
		random_shuffle(tests.begin(), tests.end());
		std::vector<std::pair <std::vector<float>, std::vector<float> > > tests_noise = tests;
		/*for (int i = 0; i < tests_noise.size(); i++)
		{
			for (int j = 0; j < tests_noise[i].first.size(); j++)
			{
				tests_noise[i].first[j] += ((max_w - min_w) * ((float)rand() / (float)RAND_MAX) + min_w);
			}
		}*/
		float* tests_h = (float*)malloc(tests_noise.size() * tests_noise[0].first.size() * sizeof(float));
		float* tests_anwsers_h = (float*)malloc(tests_noise.size() * tests_noise[0].second.size() * sizeof(float));
		for (int i = 0; i < tests_noise.size(); i++)
		{
			copy(tests_noise[i].first.begin(), tests_noise[i].first.end(), tests_h + i * tests_noise[0].first.size());
			copy(tests_noise[i].second.begin(), tests_noise[i].second.end(), tests_anwsers_h + i * tests_noise[0].second.size());
		}
		float* tests_d;
		float* tests_anwsers_d;
		cudaMalloc(&tests_d, tests_noise.size() * tests_noise[0].first.size() * sizeof(float));
		cudaMalloc(&tests_anwsers_d, tests_noise.size() * tests_noise[0].second.size() * sizeof(float));
		cudaMemcpy(tests_d, tests_h, tests_noise.size() * tests_noise[0].first.size() * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(tests_anwsers_d, tests_anwsers_h, tests_noise.size() * tests_noise[0].second.size() * sizeof(float),
			cudaMemcpyHostToDevice);
		thrust::device_vector<float> errors_d(tests_noise.size());
		for (int i = 0; i < tests_noise.size(); i++)
		{
			forward_pass(tests_d, i, tests_noise[0].first.size());
			backward_pass(tests_anwsers_d, i, tests_noise[0].second.size(), thrust::raw_pointer_cast(&errors_d[0]));
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
	printf("time: %f\n\n", (float)time / CLOCKS_PER_SEC);
	cudaFree(devStates);
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
	thrust::host_vector<float> variance_h = variance;
	for (int i = 0; i < variance_h.size(); i++)
	{
		fprintf(f, "%f ", variance_h[i]);
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

__global__ void calculate_first_layer(layer* layers, float* test, float* average, float* variance, int id, int size)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	layers[0].outputs[j] = test[id * size + j] - average[j];
 	if (variance[j] > 1.0)
 	{
		layers[0].outputs[j] /= (1.0 * variance[j]);
 	}
}

__global__ void calculate_layer(layer* layers, int i, float alpha, float beta)
{
	matrix_mul_gpu(layers[i - 1].outputs, layers[i].weights, layers[i].outputs, 1, layers[i].inputs, layers[i].size);
	__syncthreads();
	matrix_mul_gpu(layers[i].outputs, (float) (1.0 / layers[i].inputs), layers[i].outputs, 1, layers[i].size);
	__syncthreads();
	matrix_add_gpu(layers[i].outputs, layers[i].borders, layers[i].outputs, 1, layers[i].size);
	__syncthreads();
	matrix_func_gpu(layers[i].outputs, alpha, beta, layers[i].size);
}

void neural_network::forward_pass(float* tests, int id, int size)
{
	dim3 block(BLOCK_SIZE, 1);
	dim3 grid(1 + size / (1 + BLOCK_SIZE), 1);
	calculate_first_layer<<<grid, block>>>(thrust::raw_pointer_cast(&layers[0]), tests, thrust::raw_pointer_cast(&average[0]),
		thrust::raw_pointer_cast(&variance[0]), id, size);
	cudaDeviceSynchronize();
	block.y = BLOCK_SIZE;
	grid.x = 1 + max_dim / (1 + BLOCK_SIZE);
	for (int i = 1; i < depth; i++)
	{
		calculate_layer<<<grid, block>>>(thrust::raw_pointer_cast(&layers[0]), i, alpha, beta);
		cudaDeviceSynchronize();
	}
}

__global__ void calculate_last_layer(layer* layers, float* tests_anwsers, int depth, float alpha, float beta, float momentum,
	 float learning_speed, float* errors, int id, int size, int tests_size, float lambda)
{
	errors[id] = 0;
	for (int i = 0; i < size; i++)
	{
		float curr_out = layers[depth - 1].outputs[i];
		errors[id] += (tests_anwsers[id * size + i] - curr_out) * (tests_anwsers[id * size + i] - curr_out);
		layers[depth - 1].deltas[i] = (tests_anwsers[id * size + i] - curr_out) * (alpha - curr_out) * (alpha + curr_out) * beta / alpha;  
		for (int j = 0; j < layers[depth - 1].inputs; j++)
		{
			layers[depth - 1].delta_weights[j * layers[depth - 1].size + i] = momentum *
				layers[depth - 1].delta_weights[j * layers[depth - 1].size + i] + learning_speed * (layers[depth - 1].deltas[i] *
				layers[depth - 2].outputs[j] + lambda * 2.0 / tests_size * 
				layers[depth - 1].weights[j * layers[depth - 1].size + i]); 
		}
	}
	errors[id] /= 2;
}

__global__ void calculate_layer(layer* layers, float* weights_transposed, float* deltas_mul_weights, float* outputs_der,
	 float* outputs_mul_deltas, float* weights_mul_lambda, float alpha, float beta, float momentum, float learning_speed, int i,
	 int tests_size, float lambda)
{
	matrix_transpose_gpu(layers[i + 1].weights, weights_transposed, layers[i + 1].inputs, layers[i + 1].size);
	__syncthreads();
	matrix_mul_gpu(layers[i + 1].deltas, weights_transposed, deltas_mul_weights, 1, layers[i + 1].size, layers[i + 1].inputs);
	__syncthreads();
	matrix_func_der_gpu(layers[i].outputs, outputs_der, alpha, beta, layers[i].size);
	__syncthreads();
	matrix_mul_diagonal_gpu(outputs_der, deltas_mul_weights, layers[i].deltas, layers[i + 1].inputs);
	__syncthreads();
	matrix_mul_gpu(layers[i].delta_weights, momentum, layers[i].delta_weights, layers[i].inputs, layers[i].size);
	__syncthreads();
	matrix_mul_gpu(layers[i - 1].outputs, layers[i].deltas, outputs_mul_deltas, layers[i].inputs, 1, layers[i].size);
	__syncthreads();
	matrix_mul_gpu(layers[i].weights, 2.0 * lambda / tests_size, weights_mul_lambda, layers[i].inputs, layers[i].size);
	__syncthreads();
	matrix_add_gpu(weights_mul_lambda, outputs_mul_deltas, outputs_mul_deltas, layers[i].inputs, layers[i].size);
	__syncthreads();
	matrix_mul_gpu(outputs_mul_deltas, learning_speed, outputs_mul_deltas, layers[i].inputs, layers[i].size);
	__syncthreads();
	matrix_add_gpu(layers[i].delta_weights, outputs_mul_deltas, layers[i].delta_weights, layers[i].inputs, layers[i].size);
}

__global__ void update_weights(layer* layers, int i)
{
	matrix_add_gpu(layers[i].weights, layers[i].delta_weights, layers[i].weights, layers[i].inputs, layers[i].size);
}

void neural_network::backward_pass(float* tests_anwsers, int id, int size, float* errors)
{
	dim3 block(1, 1);
	dim3 grid(1, 1);
	calculate_last_layer<<<grid, block>>>(thrust::raw_pointer_cast(&layers[0]), tests_anwsers, depth, alpha, beta,
		momentum, learning_speed, errors, id, size, tests_size, lambda);
	cudaDeviceSynchronize(); 
	block.x = BLOCK_SIZE;
	block.y = BLOCK_SIZE;
	grid.x = 1 + max_dim / (1 + BLOCK_SIZE);
	grid.y = 1 + max_dim / (1 + BLOCK_SIZE);
	for (int i = depth - 2; i > 0; i--)
	{
		float local_learning_speed = learning_speed + (learning_add * (depth - i - 1));
		float* weights_transposed;
		float* deltas_mul_weights;
		float* outputs_der;
		float* outputs_mul_deltas;
		float* weights_mul_lambda;
		cudaMalloc(&weights_transposed, outputs * hidden_layer_size * sizeof(float));
		cudaMalloc(&deltas_mul_weights, hidden_layer_size * sizeof(float));
		cudaMalloc(&outputs_der, hidden_layer_size * sizeof(float));
		cudaMalloc(&outputs_mul_deltas, hidden_layer_size * inputs * sizeof(float));
		cudaMalloc(&weights_mul_lambda, hidden_layer_size * inputs * sizeof(float));
		calculate_layer<<<grid, block>>>(thrust::raw_pointer_cast(&layers[0]),
			weights_transposed, deltas_mul_weights, outputs_der, outputs_mul_deltas, weights_mul_lambda, alpha, beta,
			momentum, local_learning_speed, i, tests_size, lambda);
		cudaDeviceSynchronize();
		cudaFree(weights_transposed);
		cudaFree(deltas_mul_weights);
		cudaFree(outputs_der);
		cudaFree(outputs_mul_deltas);
		cudaFree(weights_mul_lambda);
	}
	for (int i = 1; i < depth; i++)
	{
		update_weights<<<grid, block>>>(thrust::raw_pointer_cast(&layers[0]), i);
		cudaDeviceSynchronize();
	}
}

void neural_network::normalize_old(std::vector<std::pair <std::vector<float>, std::vector<float> > > const& tests,
	float max_val, float min_freq)
{
	int n = tests[0].first.size();
	variance.resize(n, 1);
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
			variance[i] = 1.0f * sum[i] / freq[i];
		}
	}
}

void neural_network::normalize(std::vector<std::pair <std::vector<float>, std::vector<float> > > const& tests)
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

__global__ void setup_curand(curandState* state, unsigned long seed)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &state[id]);
} 

__global__ void generate_random(curandState* globalState, layer* layers, int i) 
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < layers[i].size)
	{
		layers[i].borders[id] = curand_normal(&globalState[id]);
	}
}

void neural_network::make_noise_init(curandState* devStates)
{
	dim3 block(BLOCK_SIZE, 1);
	dim3 grid(1 + max_dim / (1 + BLOCK_SIZE), 1);
	setup_curand<<<grid, block>>>(devStates, time(NULL));
	cudaDeviceSynchronize();
}

void neural_network::make_noise(curandState* devStates)
{
	dim3 block(BLOCK_SIZE, 1);
	dim3 grid(1 + max_dim / (1 + BLOCK_SIZE), 1);
	for (int i = 0; i < depth; i++)
	{
		generate_random<<<grid, block>>>(devStates, thrust::raw_pointer_cast(&layers[0]), i);
		cudaDeviceSynchronize();
	}
}

