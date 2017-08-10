#include <png++/png.hpp>
#include <fstream>
#include <vector>
#include <algorithm>
#define NOMINMAX 
#include <Windows.h>

template <typename T>
void Transpose(T* src, T* dst, const int N, const int M) {
	for (int n = 0; n<N*M; n++) {
		int i = n / N;
		int j = n%N;
		dst[n] = src[M*j + i];
	}
}

int main()
{
	const int width = 2048;
	const int height = 2048;
	const int num_frames = 20;
	std::ifstream binfile("D:/out_stream.bin", std::ios::in | std::ios::binary);
	std::vector<float> buf(width * height * num_frames);
	binfile.read((char*)buf.data(), width * height * num_frames * sizeof(float));
	std::vector<float> frame_avg(num_frames);
	for (int i = 0; i < num_frames; ++i) {
		float min_pix = buf[width * height * i];
		float max_pix = buf[width * height * i];
		for (int j = 0; j < height * width; ++j) {
			min_pix = std::min(min_pix, buf[j + height * width * i]);
			max_pix = std::max(max_pix, buf[j + height * width * i]);
		}
		for (int j = 0; j < height * width; ++j) {
			buf[j + height * width * i] = 1.0f - (buf[j + height * width * i] - min_pix) / (max_pix - min_pix);
		}
	}
	for (int i = 0; i < num_frames; ++i) {
		png::image<png::gray_pixel> img(width, height);
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				img.set_pixel(x, y, buf[x / 2 + width * (y + height * i)] * 255);
			}
		}
		img.write("D:/out_img" + std::to_string(i) + ".png");
	}
    return 0;
}

