#include <png++/png.hpp>
#include <fstream>
#include <vector>

int main()
{
	std::ifstream binfile("D:/out_stream.bin", std::ios::in | std::ios::binary);
	std::vector<float> buf(335544320 / sizeof(float));
	binfile.read((char*)buf.data(), 335544320);
	png::image<png::gray_pixel> img(2048, 335544320 / (sizeof(float) * 2048));
	for (int y = 0; y < 335544320 / (sizeof(float) * 2048); ++y) {
		double avg = 0;
		for (int x = 0; x < 2048; ++x) {
			avg += buf[2048 * y + x];
		}
		avg = avg / 2048;
		for (int x = 0; x < 2048; ++x) {
			//img.set_pixel(x, y, buf[(2050 * y + x < buf.size()) ? (2050 * y + x) : (buf.size() - 1)] / avg * 255);
			img.set_pixel(x, y, buf[2048 * y + x / 2] / avg * 255);
		}
	}
	img.write("D:/out_img.png");
    return 0;
}

