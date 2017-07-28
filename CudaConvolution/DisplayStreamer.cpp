#include "DisplayStreamer.h"
#include <stdexcept>
#include <thread>
#include <Windows.h>
#include <gdiplus.h>

const char* DisplayWindowClass = "DisplayWindowClassName";

DisplayStreamer::DisplayStreamer() : m_streaming(false), m_setup(false)
{
	WNDCLASSEX wc;
	if (!GetClassInfoEx(GetModuleHandle(NULL), DisplayWindowClass, &wc)) {
		ZeroMemory(&wc, sizeof(wc));
		wc.cbSize = sizeof(wc);
		wc.style = CS_VREDRAW | CS_HREDRAW;
		wc.lpszClassName = DisplayWindowClass;
		wc.cbWndExtra = sizeof(DisplayStreamer*);
		wc.lpfnWndProc = DisplayStreamerWinProc;
		wc.hbrBackground = WHITE_BRUSH;
		if (!RegisterClassEx(&wc)) throw std::runtime_error("Couldn't register display window class.");
	}
}


DisplayStreamer::~DisplayStreamer()
{
	StopStreaming();
}

void DisplayStreamer::Setup()
{
	m_setup = true;
}

void DisplayStreamer::StartStreaming()
{
	if (!m_setup) throw std::runtime_error("DisplayStreamer wasn't set up before its StartStreaming() was called.");
	m_streaming = true;
	m_hwnd = CreateWindow(DisplayWindowClass, "Live view", WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT, 400, 300, NULL, NULL, GetModuleHandle(NULL), this);
	m_streamthread = std::thread(MessageLoop, this);
}

void DisplayStreamer::StopStreaming()
{
	PostMessage(m_hwnd, WM_CLOSE, 0, 0);
	m_streamthread.join();
}

void DisplayStreamer::MessageLoop(DisplayStreamer* streamer)
{
	MSG msg;
	while (GetMessage(&msg, streamer->m_hwnd, 0, 0) > 0 && streamer->m_streaming) {
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
}

LRESULT DisplayStreamer::DisplayStreamerWinProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	switch (msg) {
	case WM_CREATE:
		SetWindowLongPtr(hwnd, 0, lParam);
		break;
	case WM_CLOSE:
		DestroyWindow(hwnd);
		break;
	case WM_QUIT:
		PostQuitMessage(0);
		break;
	case WM_PAINT:
	{
		DisplayStreamer* streamer = (DisplayStreamer*)GetWindowLongPtr(hwnd, 0);
		Gdiplus::Graphics g(hwnd);
		Gdiplus::Pen pen(Gdiplus::Color(0, 0, 0));
		g.SetSmoothingMode(Gdiplus::SmoothingModeAntiAlias);
		for (int i = 1; i < streamer->m_line.size(); ++i) {
			g.DrawLine(&pen, i - 1, (int)streamer->m_line[i - 1].x, i, (int)streamer->m_line[i].x);
		}
		break;
	}
	default:
		return DefWindowProc(hwnd, msg, wParam, lParam);
	}
}
