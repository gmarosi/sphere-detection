#pragma once

#include "gCamera.h"
#include "PointCloud.h"

class App
{
public:
	App();

	bool Init();
	bool InitCl();
	void Clean();

	void Update();
	void Render();

	void KeyboardDown(SDL_KeyboardEvent&);
	void KeyboardUp(SDL_KeyboardEvent&);
	void MouseMove(SDL_MouseMotionEvent&);
	void MouseDown(SDL_MouseButtonEvent&);
	void MouseUp(SDL_MouseButtonEvent&);
	void MouseWheel(SDL_MouseWheelEvent&);
	void Resize(int, int);

private:
	PointCloud* pointCloud;

	gCamera camera;

	// CL
	cl::Context context;
	cl::CommandQueue commandQueue;
};