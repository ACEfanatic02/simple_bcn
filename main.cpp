#define _CRT_SECURE_NO_WARNINGS (1)

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION (1)
#define STB_IMAGE_WRITE_IMPLEMENTATION (1)
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

#define RGBCX_IMPLEMENTATION (1)
#define RGBCX_USE_SMALLER_TABLES (1)
#include "rgbcx.h"

struct image_rgba8_t
{
	int width;
	int height;
	uint8_t* pixels;
};

image_rgba8_t load_png(const char* filename)
{
	image_rgba8_t image;
	image.pixels = stbi_load(filename, &image.width, &image.height, NULL, 4);
	if (!image.pixels)
	{
		fprintf(stderr, "Failed to load image file `%s`, missing / corrupt?", filename);
		exit(1);
	}
	return image;
}

void free_image(image_rgba8_t* img)
{
	free(img->pixels);
	memset(img, 0, sizeof(*img));
}

const int k_texels_per_block = 16;

struct bc4_block_t
{
	uint8_t endpoint0;
	uint8_t endpoint1;
	uint8_t selectors[k_texels_per_block]; // 3-bit indices
};

union bc4_packed_block_t
{
	struct
	{
		uint8_t endpoint0;
		uint8_t endpoint1;
		uint8_t packed_selectors[6];
	};
	uint64_t qword;
};
static_assert(sizeof(bc4_packed_block_t) == sizeof(uint64_t));

bc4_packed_block_t pack_bc4(const bc4_block_t* block)
{
	bc4_packed_block_t result = { 0 };

	uint64_t packed_selectors = 0;
	for (int i = 0; i < _countof(block->selectors); ++i)
	{
		assert(block->selectors[i] < 8);
		packed_selectors |= ((uint64_t)block->selectors[i] << (i * 3));
	}

	result.endpoint0 = block->endpoint0;
	result.endpoint1 = block->endpoint1;
	result.packed_selectors[0] = static_cast<uint8_t>(packed_selectors >> 0);
	result.packed_selectors[1] = static_cast<uint8_t>(packed_selectors >> 8);
	result.packed_selectors[2] = static_cast<uint8_t>(packed_selectors >> 16);
	result.packed_selectors[3] = static_cast<uint8_t>(packed_selectors >> 24);
	result.packed_selectors[4] = static_cast<uint8_t>(packed_selectors >> 32);
	result.packed_selectors[5] = static_cast<uint8_t>(packed_selectors >> 40);

	return result;
}

void encode_bc4_compute_palette8(uint8_t* out_palette, int v_max, int v_min)
{
	out_palette[0] = static_cast<uint8_t>(v_max);
	out_palette[1] = static_cast<uint8_t>(v_min);
	out_palette[2] = static_cast<uint8_t>((v_max * 6 + v_min * 1) / 7);
	out_palette[3] = static_cast<uint8_t>((v_max * 5 + v_min * 2) / 7);
	out_palette[4] = static_cast<uint8_t>((v_max * 4 + v_min * 3) / 7);
	out_palette[5] = static_cast<uint8_t>((v_max * 3 + v_min * 4) / 7);
	out_palette[6] = static_cast<uint8_t>((v_max * 2 + v_min * 5) / 7);
	out_palette[7] = static_cast<uint8_t>((v_max * 1 + v_min * 6) / 7);
}

void encode_bc4_compute_palette6(uint8_t* out_palette, int v_max, int v_min)
{
	out_palette[0] = static_cast<uint8_t>(v_min);
	out_palette[1] = static_cast<uint8_t>(v_max);
	out_palette[2] = static_cast<uint8_t>((v_min * 4 + v_max * 1) / 5);
	out_palette[3] = static_cast<uint8_t>((v_min * 3 + v_max * 2) / 5);
	out_palette[4] = static_cast<uint8_t>((v_min * 2 + v_max * 3) / 5);
	out_palette[5] = static_cast<uint8_t>((v_min * 1 + v_max * 4) / 5);
	out_palette[6] = static_cast<uint8_t>(0);
	out_palette[7] = static_cast<uint8_t>(255);
}

int encode_bc4_fit_selectors_for_palette(const uint8_t* palette_entries, const uint8_t* in_pixels, uint8_t* out_selectors)
{
	int total_sq_error = 0;
	for (int i = 0; i < k_texels_per_block; ++i)
	{
		int v = in_pixels[i];

		// Search for palette entry that minimizes the error:
		uint8_t best_selector = 0;
		int best_error = std::abs(v - (int)palette_entries[0]);
		for (uint8_t p = 0; p < 8; ++p)
		{
			int error = std::abs(v - (int)palette_entries[p]);
			if (error < best_error)
			{
				best_selector = p;
				best_error = error;
			}
		}

		out_selectors[i] = best_selector;
		total_sq_error += best_error * best_error;
	}

	return total_sq_error;
}

int encode_bc4_fit_selectors_for_endpoints8(int endpoint0, int endpoint1, const uint8_t* in_pixels, uint8_t* out_selectors)
{
	// Remap from lerp indices (min/max at ends) to selectors (min/max in first two entries.)
	// I'm sure there's a bit-twiddling way to do this, but I can't be bothered.
	static const uint8_t k_index_to_selector_lut[8] = { 1, 7, 6, 5, 4, 3, 2, 0 };

	// 8-color mode means endpoint0 is the maximum and endpoint1 is the minimum.
	assert(endpoint0 > endpoint1);
	const int range = endpoint0 - endpoint1;
	const int bias = (range < 8) ? (range - 1) : (range / 2 + 2);

	uint8_t palette[8];
	encode_bc4_compute_palette8(palette, endpoint0, endpoint1);

	int total_sq_error = 0;
	for (int i = 0; i < k_texels_per_block; ++i)
	{
		int v = in_pixels[i];

		int index = ((v - endpoint1) * 7 + bias) / range;
		int selector = k_index_to_selector_lut[std::clamp(index, 0, 7)];
		int error = v - palette[selector];

		out_selectors[i] = selector;
		total_sq_error += error * error;
	}

	return total_sq_error;
}

int encode_bc4_fit_selectors_for_endpoints6(int endpoint0, int endpoint1, const uint8_t* in_pixels, uint8_t* out_selectors)
{
	static const uint8_t k_index_to_selector_lut[6] = { 0, 2, 3, 4, 5, 1 };

	assert(endpoint0 <= endpoint1);
	const int range = std::max(endpoint1 - endpoint0, 1);
	const int bias = (range < 6) ? (range - 1) : (range / 2 + 2);

	uint8_t palette[8];
	encode_bc4_compute_palette6(palette, endpoint1, endpoint0);

	int total_sq_error = 0;
	for (int i = 0; i < k_texels_per_block; ++i)
	{
		int v = in_pixels[i];

		int index = ((v - endpoint0) * 5 + bias) / range;
		int selector = k_index_to_selector_lut[std::clamp(index, 0, 5)];
		int error = v - palette[selector];
		int error_0 = v - 0;
		int error_255 = v - 255;

		if (error_0 * error_0 < error * error)
		{
			selector = 6;
			error = error_0;
		}

		if (error_255 * error_255 < error * error)
		{
			selector = 7;
			error = error_255;
		}

		out_selectors[i] = selector;
		total_sq_error += error * error;
	}

	return total_sq_error;
}

void encode_bc4_exhaustive(bc4_block_t* out_result, const uint8_t* in_pixels)
{
	int best_error = INT_MAX;
	uint8_t temp_selectors[k_texels_per_block];

	for (int endpoint0 = 0; endpoint0 < 256; endpoint0++)
	{
		for (int endpoint1 = 0; endpoint1 < 256; endpoint1++)
		{
			if (endpoint0 > endpoint1)
			{
				int error = encode_bc4_fit_selectors_for_endpoints8(endpoint0, endpoint1, in_pixels, temp_selectors);
				if (error < best_error)
				{
					out_result->endpoint0 = static_cast<uint8_t>(endpoint0);
					out_result->endpoint1 = static_cast<uint8_t>(endpoint1);
					memcpy(out_result->selectors, temp_selectors, sizeof(temp_selectors));
					best_error = error;
				}
			}
			else
			{
				int error = encode_bc4_fit_selectors_for_endpoints6(endpoint0, endpoint1, in_pixels, temp_selectors);
				if (error < best_error)
				{
					out_result->endpoint0 = static_cast<uint8_t>(endpoint0);
					out_result->endpoint1 = static_cast<uint8_t>(endpoint1);
					memcpy(out_result->selectors, temp_selectors, sizeof(temp_selectors));
					best_error = error;
				}
			}
		}
	}
}

void encode_bc4_optimize_endpoints_least_squares8_float(
	const bc4_block_t* block,
	const uint8_t* in_pixels,
	int* out_endpoint0,
	int* out_endpoint1)
{
	static const float k_selector_to_weight[8] = 
	{ 
		7.0f / 7.0f, 
		0.0f / 7.0f, 
		6.0f / 7.0f, 
		5.0f / 7.0f, 
		4.0f / 7.0f, 
		3.0f / 7.0f, 
		2.0f / 7.0f, 
		1.0f / 7.0f 
	};

	float sum_a2 = 0.0f;
	float sum_b2 = 0.0f;
	float sum_ab = 0.0f;

	float sum_va = 0.0f;
	float sum_v = 0.0f;

	for (int i = 0; i < k_texels_per_block; ++i)
	{
		int selector = block->selectors[i];
		float a = k_selector_to_weight[selector];
		float b = 1.0f - a;

		sum_a2 += a * a;
		sum_b2 += b * b;
		sum_ab += a * b;

		float v = in_pixels[i] / 255.0f;
		sum_va += v * a;
		sum_v += v;
	}
	float sum_vb = sum_v - sum_va;

	float det = sum_a2 * sum_b2 - sum_ab * sum_ab;
	if (fabsf(det) > 1e-7f)
	{
		float rcp_det = 1.0f / det;

		float ep0 = (sum_va * sum_b2 - sum_ab * sum_vb) * rcp_det;
		float ep1 = (sum_vb * sum_a2 - sum_ab * sum_va) * rcp_det;

		ep0 = std::clamp(255.0f * ep0, 0.0f, 255.0f);
		ep1 = std::clamp(255.0f * ep1, 0.0f, 255.0f);

		*out_endpoint0 = static_cast<int>(ep0 + 0.5f);
		*out_endpoint1 = static_cast<int>(ep1 + 0.5f);
	}
	else
	{
		*out_endpoint0 = block->endpoint0;
		*out_endpoint1 = block->endpoint1;
	}
}

void encode_bc4_optimize_endpoints_least_squares6_float(
	const bc4_block_t* block,
	const uint8_t* in_pixels,
	int* out_endpoint0,
	int* out_endpoint1)
{
	static const float k_selector_to_weight[6] =
	{
		5.0f / 5.0f,
		0.0f / 5.0f,
		4.0f / 5.0f,
		3.0f / 5.0f,
		2.0f / 5.0f,
		1.0f / 5.0f,
	};

	float sum_a2 = 0.0f;
	float sum_b2 = 0.0f;
	float sum_ab = 0.0f;

	float sum_va = 0.0f;
	float sum_v = 0.0f;

	for (int i = 0; i < k_texels_per_block; ++i)
	{
		int selector = block->selectors[i];
		if (selector > 5)
		{
			continue;  // XX:  Ignore constant selectors
		}

		float a = k_selector_to_weight[selector];
		float b = 1.0f - a;

		sum_a2 += a * a;
		sum_b2 += b * b;
		sum_ab += a * b;

		float v = in_pixels[i] / 255.0f;
		sum_va += v * a;
		sum_v += v;
	}
	float sum_vb = sum_v - sum_va;

	float det = sum_a2 * sum_b2 - sum_ab * sum_ab;
	if (fabsf(det) > 1e-7f)
	{
		float rcp_det = 1.0f / det;

		float ep0 = (sum_va * sum_b2 - sum_ab * sum_vb) * rcp_det;
		float ep1 = (sum_vb * sum_a2 - sum_ab * sum_va) * rcp_det;

		ep0 = std::clamp(255.0f * ep0, 0.0f, 255.0f);
		ep1 = std::clamp(255.0f * ep1, 0.0f, 255.0f);

		*out_endpoint0 = static_cast<int>(ep0 + 0.5f);
		*out_endpoint1 = static_cast<int>(ep1 + 0.5f);
	}
	else
	{
		*out_endpoint0 = block->endpoint0;
		*out_endpoint1 = block->endpoint1;
	}
}

void encode_bc4_optimize_endpoints_least_squares(
	const bc4_block_t* block, 
	const uint8_t* in_pixels, 
	int* out_endpoint0, 
	int* out_endpoint1)
{
	if (block->endpoint0 > block->endpoint1)
	{
		encode_bc4_optimize_endpoints_least_squares8_float(block, in_pixels, out_endpoint0, out_endpoint1);
	}
	else
	{
		encode_bc4_optimize_endpoints_least_squares6_float(block, in_pixels, out_endpoint0, out_endpoint1);
	}
}

int encode_bc4_optimize_block(bc4_block_t* inout_block, const uint8_t* in_pixels, int best_error)
{
	static const int k_max_ls_iterations = 16;
	uint8_t temp_selectors[k_texels_per_block];

	if (inout_block->endpoint0 > inout_block->endpoint1)
	{
		for (int ls_iter = 0; ls_iter < k_max_ls_iterations; ++ls_iter)
		{
			int ls_ep0;
			int ls_ep1;
			encode_bc4_optimize_endpoints_least_squares8_float(inout_block, in_pixels, &ls_ep0, &ls_ep1);

			int ls_error = encode_bc4_fit_selectors_for_endpoints8(ls_ep0, ls_ep1, in_pixels, temp_selectors);
			if (ls_error < best_error)
			{
				inout_block->endpoint0 = static_cast<uint8_t>(ls_ep0);
				inout_block->endpoint1 = static_cast<uint8_t>(ls_ep1);

				memcpy(inout_block->selectors, temp_selectors, sizeof(temp_selectors));
				best_error = ls_error;
			}
			else
			{
				// Did not improve error, stop search.
				break;
			}
		}
	}
	else
	{
		for (int ls_iter = 0; ls_iter < k_max_ls_iterations; ++ls_iter)
		{
			int ls_ep0;
			int ls_ep1;
			encode_bc4_optimize_endpoints_least_squares6_float(inout_block, in_pixels, &ls_ep0, &ls_ep1);

			int ls_error = encode_bc4_fit_selectors_for_endpoints6(ls_ep0, ls_ep1, in_pixels, temp_selectors);
			if (ls_error < best_error)
			{
				inout_block->endpoint0 = static_cast<uint8_t>(ls_ep0);
				inout_block->endpoint1 = static_cast<uint8_t>(ls_ep1);

				memcpy(inout_block->selectors, temp_selectors, sizeof(temp_selectors));
				best_error = ls_error;
			}
			else
			{
				// Did not improve error, stop search.
				break;
			}
		}
	}
	return best_error;
}

void encode_bc4(bc4_block_t* out_result, const uint8_t* in_pixels)
{
	// First, we need to determine the bounds of values within this block.  Compute min and max:
	int v_min = in_pixels[0];
	int v_max = in_pixels[0];
	// Separately track min/max for 6-color mode, which excludes texel values of 0 or 255 (covered by constant selectors.)
	int v_min_6col = INT_MAX;
	int v_max_6col = INT_MIN;
	for (int i = 0; i < k_texels_per_block; ++i)
	{
		int v = in_pixels[i];
		v_min = std::min(v_min, v);
		v_max = std::max(v_max, v);

		if (v != 0 && v != 255)
		{
			v_min_6col = std::min(v_min_6col, v);
			v_max_6col = std::max(v_max_6col, v);
		}
	}
	// Handle the case where _all_ texels are 0 or 255.  Need to keep 6-color mode endpoints in range:
	if (v_min_6col > 255) { v_min_6col = v_min; }
	if (v_max_6col < 0) { v_max_6col = v_max; }

	// Special case:  v_min == v_max.  Block is a flat color.  We can early out here:
	if (v_min == v_max)
	{
		out_result->endpoint0 = static_cast<uint8_t>(v_max);
		out_result->endpoint1 = static_cast<uint8_t>(v_min);
		// Select endpoint 0 for all texels in block.
		// NOTE:  For BC4/BC5 only (_not_ BC3 alpha), values are decoded at higher precision (14-16 bits 
		// instead of 8.)  If you have high precision inputs, could make sense to use interpolated
		// selectors even in flat blocks to increase the output precision.
		memset(out_result->selectors, 0, sizeof(out_result->selectors));
		return;
	}

	// Palette choice:
	// By swapping the order of endpoints, we can encode with one of two palettes.
	// 
	// interp8, ep0 > ep1					|| interp6, ep0 <= ep1
	//										||
	// 000 : ep0							|| 000 : ep0
	// 001 : ep1							|| 001 : ep1
	// 010 : (ep0*6 + ep1*1) / 7			|| 010 : (ep0*4 + ep1*1) / 5
	// 011 : (ep0*5 + ep1*2) / 7			|| 011 : (ep0*3 + ep1*2) / 5
	// 100 : (ep0*4 + ep1*3) / 7			|| 100 : (ep0*2 + ep1*3) / 5
	// 101 : (ep0*3 + ep1*4) / 7			|| 101 : (ep0*1 + ep1*4) / 5
	// 110 : (ep0*2 + ep1*5) / 7			|| 110 : 0
	// 111 : (ep0*1 + ep1*6) / 7			|| 111 : 255
	//
	// The second mode can be useful to improve precision when values in a block 
	// span a small range except for a few extreme texels.  However, you can still 
	// get high quality results in most cases without it.

	bc4_block_t trial_block8 = { static_cast<uint8_t>(v_max), static_cast<uint8_t>(v_min) };
	int interp8_error = encode_bc4_fit_selectors_for_endpoints8(trial_block8.endpoint0, trial_block8.endpoint1, in_pixels, trial_block8.selectors);
	interp8_error = encode_bc4_optimize_block(&trial_block8, in_pixels, interp8_error);

	bc4_block_t trial_block6 = { static_cast<uint8_t>(v_min_6col), static_cast<uint8_t>(v_max_6col) };
	int interp6_error = encode_bc4_fit_selectors_for_endpoints6(trial_block6.endpoint0, trial_block6.endpoint1, in_pixels, trial_block6.selectors);
	interp6_error = encode_bc4_optimize_block(&trial_block6, in_pixels, interp6_error);

	if (interp8_error < interp6_error)
	{
		*out_result = trial_block8;
	}
	else
	{
		*out_result = trial_block6;
	}
}

struct image_compression_test_result_t
{
	uint64_t elapsed_time_ns;
	double rmse;

	uint64_t rgbcx_elapsed_time_ns;
	double rgbcx_rmse;
};

image_compression_test_result_t test_compression_performance_bc4(const char* filename)
{
	image_compression_test_result_t result = { 0 };

	image_rgba8_t image = load_png(filename);

	const bool enable_exhaustive_point_test = false;
	const int64_t block_count_x = image.width / 4;
	const int64_t block_count_y = image.height / 4;
	const int64_t channels = 1;

	bc4_packed_block_t* encoded_blocks = (bc4_packed_block_t*)calloc(block_count_x * block_count_y * channels, sizeof(*encoded_blocks));
	bc4_packed_block_t* rgbcx_encoded_blocks = (bc4_packed_block_t*)calloc(block_count_x * block_count_y * channels, sizeof(*encoded_blocks));
	assert(encoded_blocks != NULL && rgbcx_encoded_blocks != NULL);

	auto encode_start_time = std::chrono::steady_clock::now();
	for (int block_y = 0; block_y < block_count_y; ++block_y)
	{
		for (int block_x = 0; block_x < block_count_x; ++block_x)
		{
			for (int channel = 0; channel < channels; ++channel)
			{
				uint8_t texels[16] = { 0 };
				for (int by = 0; by < 4; ++by)
				{
					for (int bx = 0; bx < 4; ++bx)
					{
						int ix = block_x * 4 + bx;
						int iy = block_y * 4 + by;
						texels[by * 4 + bx] = image.pixels[(iy * image.width + ix) * 4 + channel];
					}
				}
				bc4_block_t unpacked_result;
				encode_bc4(&unpacked_result, texels);

				int64_t encoded_block_index = (block_y * block_count_x + block_x) * channels + channel;
				encoded_blocks[encoded_block_index] = pack_bc4(&unpacked_result);
			}
		}
	}
	auto encode_end_time = std::chrono::steady_clock::now();
	auto encode_duration = encode_end_time - encode_start_time;

	auto rgbcx_encode_start_time = std::chrono::steady_clock::now();
	for (int block_y = 0; block_y < block_count_y; ++block_y)
	{
		for (int block_x = 0; block_x < block_count_x; ++block_x)
		{
			for (int channel = 0; channel < channels; ++channel)
			{
				uint8_t texels[16] = { 0 };
				for (int by = 0; by < 4; ++by)
				{
					for (int bx = 0; bx < 4; ++bx)
					{
						int ix = block_x * 4 + bx;
						int iy = block_y * 4 + by;
						texels[by * 4 + bx] = image.pixels[(iy * image.width + ix) * 4 + channel];
					}
				}
				bc4_packed_block_t rgbcx_result;
				rgbcx::encode_bc4(&rgbcx_result, texels, 1);

				int64_t encoded_block_index = (block_y * block_count_x + block_x) * channels + channel;
				rgbcx_encoded_blocks[encoded_block_index] = rgbcx_result;
			}
		}
	}
	auto rgbcx_encode_end_time = std::chrono::steady_clock::now();
	auto rgbcx_encode_duration = rgbcx_encode_end_time - rgbcx_encode_start_time;

	result.elapsed_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(encode_duration).count();
	result.rgbcx_elapsed_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(rgbcx_encode_duration).count();

	double sum_sq_error = 0.0;
	double rgbcx_sum_sq_error = 0.0;
	for (int block_y = 0; block_y < block_count_y; ++block_y)
	{
		for (int block_x = 0; block_x < block_count_x; ++block_x)
		{
			for (int channel = 0; channel < channels; ++channel)
			{
				uint8_t texels[16] = { 0 };
				for (int by = 0; by < 4; ++by)
				{
					for (int bx = 0; bx < 4; ++bx)
					{
						int ix = block_x * 4 + bx;
						int iy = block_y * 4 + by;
						texels[by * 4 + bx] = image.pixels[(iy * image.width + ix) * 4 + channel];
					}
				}
				int64_t encoded_block_index = (block_y * block_count_x + block_x) * channels + channel;

				uint8_t unpacked_texels[16] = { 0 };
				if (enable_exhaustive_point_test && ((encoded_block_index & 511) == 0))
				{
					bc4_block_t exhaustive_block;
					encode_bc4_exhaustive(&exhaustive_block, texels);
					bc4_packed_block_t exhaustive_packed_block = pack_bc4(&exhaustive_block);

					rgbcx::unpack_bc4(&exhaustive_packed_block, unpacked_texels, 1);

					double exhaustive_error = 0.0;
					for (int i = 0; i < k_texels_per_block; ++i)
					{
						int delta = static_cast<int>(texels[i]) - static_cast<int>(unpacked_texels[i]);
						exhaustive_error += static_cast<double>(delta * delta);
					}
					exhaustive_error = sqrt(exhaustive_error / static_cast<double>(k_texels_per_block));

					printf("Exhaustive Block (%d, %d, %d):\n\trmse = %f\n\tendpoints = ( %d, %d )\n\tencoder endpoints = ( %d, %d )\nvalues:\n"
						"%3d %3d %3d %3d\n"
						"%3d %3d %3d %3d\n"
						"%3d %3d %3d %3d\n"
						"%3d %3d %3d %3d\n\n",
						block_x, block_y, channel,
						exhaustive_error,
						exhaustive_block.endpoint0, exhaustive_block.endpoint1,
						encoded_blocks[encoded_block_index].endpoint0, encoded_blocks[encoded_block_index].endpoint1,
						texels[ 0], texels[ 1], texels[ 2], texels[ 3],
						texels[ 4], texels[ 5], texels[ 6], texels[ 7],
						texels[ 8], texels[ 9], texels[10], texels[11],
						texels[12], texels[13], texels[14], texels[15]);
				}

				rgbcx::unpack_bc4(&encoded_blocks[encoded_block_index], unpacked_texels, 1);
				for (int i = 0; i < k_texels_per_block; ++i)
				{
					int delta = static_cast<int>(texels[i]) - static_cast<int>(unpacked_texels[i]);
					sum_sq_error += static_cast<double>(delta * delta);
				}

				rgbcx::unpack_bc4(&rgbcx_encoded_blocks[encoded_block_index], unpacked_texels, 1);
				for (int i = 0; i < k_texels_per_block; ++i)
				{
					int delta = static_cast<int>(texels[i]) - static_cast<int>(unpacked_texels[i]);
					rgbcx_sum_sq_error += static_cast<double>(delta * delta);
				}
			}
		}
	}

	result.rmse = sqrt(sum_sq_error / (image.width * image.height * channels));
	result.rgbcx_rmse = sqrt(rgbcx_sum_sq_error / (image.width * image.height * channels));

	free(encoded_blocks);
	free(rgbcx_encoded_blocks);
	free_image(&image);

	return result;
}

const char* k_test_normal_file_names[] = {
	"data/src_normals/arch_stone_wall_01_Normal.png",
	"data/src_normals/brickwall_01_Normal.png",
	"data/src_normals/brickwall_02_Normal.png",
	"data/src_normals/wood_tile_01_Normal.png",
	"data/src_normals/curtain_fabric_Normal.png",
	"data/src_normals/lionhead_01_Normal.png",
	"data/src_normals/ceiling_plaster_01_Normal.png",
	"data/src_normals/ceiling_plaster_02_Normal.png",
	"data/src_normals/col_1stfloor_Normal.png",
	"data/src_normals/col_brickwall_01_Normal.png",
	"data/src_normals/col_head_1stfloor_Normal.png",
	"data/src_normals/col_head_2ndfloor_02_Normal.png",
	"data/src_normals/col_head_2ndfloor_03_Normal.png",
	"data/src_normals/door_stoneframe_01_Normal.png",
	"data/src_normals/door_stoneframe_02_Normal.png",
	"data/src_normals/floor_tiles_01_Normal.png",
};

int main(int argc, char** argv)
{
	rgbcx::init();
	double sum_duration_ms = 0.0;
	double sum_rgbcx_duration_ms = 0.0;
	double sum_rmse = 0.0;
	double sum_rgbcx_rmse = 0.0;
	for (int i = 0; i < _countof(k_test_normal_file_names); ++i)
	{
		image_compression_test_result_t test_result = test_compression_performance_bc4(k_test_normal_file_names[i]);
		double test_duration_ms = static_cast<double>(test_result.elapsed_time_ns) * 1e-6;
		double rgbcx_duration_ms = static_cast<double>(test_result.rgbcx_elapsed_time_ns) * 1e-6;

		sum_duration_ms += test_duration_ms;
		sum_rgbcx_duration_ms += rgbcx_duration_ms;
		sum_rmse += test_result.rmse;
		sum_rgbcx_rmse += test_result.rgbcx_rmse;

		printf("[%s]\n\tTest     \trgbcx   \tRatio x100 (lower = better)\nTime:\t% 2.2fms\t% 2.2fms\t%.2f\nRMSE:\t%f\t%f\t%.2f\n\n", 
			k_test_normal_file_names[i], 
			test_duration_ms, rgbcx_duration_ms, 100.0 * test_duration_ms / rgbcx_duration_ms,
			test_result.rmse, test_result.rgbcx_rmse, 100.0 * test_result.rmse / test_result.rgbcx_rmse);
	}
	printf("Totals:\n\tTest     \trgbcx   \tRatio x100 (lower = better)\nTime:\t% 2.2fms\t% 2.2fms\t%.2f\nRMSE:\t%f\t%f\t%.2f\n\n",
		sum_duration_ms, sum_rgbcx_duration_ms, 100.0 * sum_duration_ms / sum_rgbcx_duration_ms,
		sum_rmse, sum_rgbcx_rmse, 100.0 * sum_rmse / sum_rgbcx_rmse);

	return 0;
}