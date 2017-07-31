#include <SapClassBasic.h>
#include <Windows.h>
#include <cstdio>
#include <string>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdint>
#include <mutex>
#include "cuda_runtime.h"
#include "cufft.h"

const char* config_file = "C:/Users/SD-OCT/Desktop/TargetCam.ccf";
const char* out_file = "D:/out.bin";

struct acq_context {
	std::vector<uint16_t>* data;
	int num_pix;
	int num_bytes;
	std::ofstream* fout;
	SapBufferWithTrash* buf;
	cufftComplex* output_dbuf;
	cufftComplex* output_hbuf;
	cufftReal* input_hbuf;
	cufftReal* input_dbuf;
	cufftHandle plan;
	std::mutex donttouch;
};

void transfer_callback(SapXferCallbackInfo* info)
{
	acq_context* ctx = (acq_context*)info->GetContext();
	std::lock_guard<std::mutex> lock(ctx->donttouch);
	void* pdata;
	ctx->buf->GetAddress(&pdata);
	for (int i = 0; i < ctx->num_pix; ++i) {
		ctx->input_hbuf[i] = ((uint16_t*)pdata)[i];
	}
	ctx->buf->ReleaseAddress(pdata);
	cudaMemcpy(ctx->input_dbuf, ctx->input_hbuf, ctx->num_bytes * 2, cudaMemcpyHostToDevice);
	cufftExecR2C(ctx->plan, ctx->input_dbuf, ctx->output_dbuf);
	cudaDeviceSynchronize();
	cudaMemcpy(ctx->output_hbuf, ctx->output_dbuf, ctx->num_bytes * 4, cudaMemcpyDeviceToHost);
	ctx->fout->write((char*)ctx->output_hbuf, ctx->num_bytes * 4);
}

void acquire()
{
	std::vector<uint16_t> data;
	std::ofstream fout(out_file, std::ios::out | std::ios::binary);
	char name[CORSERVER_MAX_STRLEN];
	int server_count = SapManager::GetServerCount();
	for (int i = 0; i < server_count; ++i) {
		SapManager::GetServerName(i, name, sizeof(name));
		std::printf("%d: %s\n", i, name);
	}
	std::cout << "Enter server number: ";
	int server = -1;
	std::cin >> server;
	SapManager::GetServerName(server, name, sizeof(name));
	SapLocation loc(name, server);
	std::cout << "Server " << server << ": " << name << std::endl;

	SapAcquisition acq(loc, config_file);
	SapBufferWithTrash buffer(4, &acq);
	acq_context ctx;
	SapTransfer acq_to_buf(transfer_callback, &ctx);
	acq_to_buf.AddPair(SapXferPair(&acq, &buffer));

	if (!acq.Create()) {
		MessageBox(NULL, "Could not create SapAcquisition.\nPress OK to quit.", "Error", MB_OK | MB_ICONERROR);
		goto FreeHandles;
	}
	if (!buffer.Create()) {
		MessageBox(NULL, "Could not create SapBufferWithTrash.\nPress OK to quit.", "Error", MB_OK | MB_ICONERROR);
		goto FreeHandles;
	}
	if (!acq_to_buf.Create()) {
		MessageBox(NULL, "Could not create SapTransfer.\nPress OK to quit.", "Error", MB_OK | MB_ICONERROR);
		goto FreeHandles;
	}
	ctx.data = &data;
	ctx.buf = &buffer;
	ctx.fout = &fout;
	ctx.num_pix = buffer.GetWidth() * buffer.GetHeight();
	ctx.num_bytes = ctx.num_pix * sizeof(uint16_t);
	cudaMalloc(&ctx.input_dbuf, ctx.num_pix * sizeof(cufftReal));
	cudaMalloc(&ctx.output_dbuf, ctx.num_pix * sizeof(cufftComplex));
	ctx.input_hbuf = new cufftReal[ctx.num_pix];
	ctx.output_hbuf = new cufftComplex[ctx.num_pix];
	cufftPlan1d(&ctx.plan, buffer.GetWidth(), CUFFT_R2C, buffer.GetHeight());
	data.resize(buffer.GetWidth() * buffer.GetHeight() * buffer.GetBytesPerPixel() / sizeof(uint16_t));
	if (!acq_to_buf.Grab()) {
		MessageBox(NULL, "Could not start image acquisition.\nPress OK to quit.", "Error", MB_OK | MB_ICONERROR);
		goto FreeHandles;
	}
	MessageBox(NULL, "Press OK to stop grab", "grabbing", MB_OK);
	if (!acq_to_buf.Freeze()) {
		MessageBox(NULL, "Could not stop safely.\nPress OK to quit.", "Error", MB_OK | MB_ICONERROR);
		goto FreeHandles;
	}
	if (!acq_to_buf.Wait(5000)) {
		MessageBox(NULL, "Timed out!\nPress OK to quit.", "Error", MB_OK | MB_ICONERROR);
		goto FreeHandles;
	}
	std::cout << "a\n";

FreeHandles:
	std::lock_guard<std::mutex> lock(ctx.donttouch);
	acq_to_buf.Abort();
	std::cout << "a\n";
	acq.UnregisterCallback();
	std::cout << "a\n";
	cufftDestroy(ctx.plan);
	std::cout << "a\n";
	cudaFree(ctx.input_dbuf);
	std::cout << "a\n";
	cudaFree(ctx.output_dbuf);
	std::cout << "a\n";
	delete[] ctx.input_hbuf;
	std::cout << "a\n";
	delete[] ctx.output_hbuf;
	std::cout << "a\n";
	acq_to_buf.Destroy();
	std::cout << "a\n";
	buffer.Destroy();
	std::cout << "a\n";
	acq.Destroy();
	std::cout << "a\n";
	fout.close();
	std::cout << "a\n";
}