#include "FrameGrabStreamer.h"
#include <SapClassBasic.h>
#include <stdexcept>
#include <iostream>

void FrameGrabStreamer::FrameCallback(SapXferCallbackInfo* info)
{
	FrameGrabStreamer* streamer = (FrameGrabStreamer*)info->GetContext();
	FrameGrabStreamer::Producer_element_t* buf = streamer->m_prod_in->front();
	if (streamer->m_prod_in->size() <= 0) return;
	streamer->m_prod_in->pop_front();
	void* pdata;
	streamer->m_buffer.GetAddress(&pdata);
	memcpy(buf, pdata, streamer->m_buf_size);
	streamer->m_buffer.ReleaseAddress(pdata);
	streamer->m_prod_out->push_back(buf);
}

FrameGrabStreamer::FrameGrabStreamer() : m_set_up(false)
{
	
}


FrameGrabStreamer::~FrameGrabStreamer()
{
	StopStreaming();
	m_acq_to_buf.Abort();
	m_acq.UnregisterCallback();
	m_acq_to_buf.Destroy();
	m_buffer.Destroy();
	m_acq.Destroy();
}

int FrameGrabStreamer::GetFrameWidth()
{
	if (!m_set_up) throw std::runtime_error("FrameGrabStreamer wasn't set up before calling GetFrameWidth.");
	return m_buffer.GetWidth();
}

int FrameGrabStreamer::GetFrameHeight()
{
	if (!m_set_up) throw std::runtime_error("FrameGrabStreamer wasn't set up before calling GetFrameHeight.");
	return m_buffer.GetHeight();
}

void FrameGrabStreamer::Setup()
{
	char name[CORSERVER_MAX_STRLEN];
	SapManager::GetServerName(m_server_id, name, sizeof(name));
	m_loc = SapLocation(name, m_server_id);
	m_acq = SapAcquisition(m_loc, m_config_file.c_str());
	m_buffer = SapBufferWithTrash(2, &m_acq);
	m_acq_to_buf = SapTransfer(FrameCallback, this);
	m_acq_to_buf.AddPair(SapXferPair(&m_acq, &m_buffer));
	if (!m_acq.Create()) {
		throw std::runtime_error("Could not create SapAcquisition.");
	}
	if (!m_buffer.Create()) {
		throw std::runtime_error("Could not create SapBufferWithTrash.");
	}
	if (!m_acq_to_buf.Create()) {
		throw std::runtime_error("Could not create SapTransfer.");
	}
	m_buf_size = sizeof(Producer_element_t) * m_buffer.GetWidth() * m_buffer.GetHeight();
	m_set_up = true;
}

void FrameGrabStreamer::StartStreaming()
{
	if (!m_set_up) throw std::runtime_error("FrameGrabStreamer wasn't set up before callign StartStreaming.");
	if (!m_acq_to_buf.Grab()) {
		throw std::runtime_error("Could not start image acquisition.");
	}
}

void FrameGrabStreamer::StopStreaming()
{
	if (!m_acq_to_buf.Freeze()) {
		throw std::runtime_error("Could not stop safely.");
	}
	if (!m_acq_to_buf.Wait(5000)) {
		throw std::runtime_error("Waiting for callback to end timed out!");
	}
}
