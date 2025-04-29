// GLEW
#include <GL/glew.h>

// SDL, OpenGL
#include <SDL.h>
#include <SDL_opengl.h>

#include <iostream>

#include "App.h"

int main(int argc, char* argv[])
{
	// Initialize SDL
	if (SDL_Init(SDL_INIT_VIDEO) == -1)
	{
		std::cerr << "[SDL] SDL initialization failed: " << SDL_GetError() << std::endl;
		return 1;
	}

	// OpenGL attribute setup
	SDL_GL_SetAttribute(SDL_GL_BUFFER_SIZE, 32);
	SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8);

	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

	// Create window
	SDL_Window* window = 0;
	window = SDL_CreateWindow(
		"LIDAR Sphere/Cylinder Detection",
		SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED,
		640,
		480,
		SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_SHOWN
	);

	if (window == 0)
	{
		std::cerr << "[SDL] Creating SDL window failed: " << SDL_GetError() << std::endl;
		return 1;
	}

	// Create OpenGL context
	SDL_GLContext context = SDL_GL_CreateContext(window);
	if (context == 0)
	{
		std::cerr << "[OpenGL] Creating OpenGL context failed: " << SDL_GetError() << std::endl;
		return 1;
	}

	// Wait for vsync
	SDL_GL_SetSwapInterval(0);

	// Start GLEW
	GLenum error = glewInit();
	if (error != GLEW_OK)
	{
		std::cerr << "[GLEW] Error while initializing GLEW!" << std::endl;
		return 1;
	}

	// Get OpenGL version
	int glVersion[2] = { -1, -1 };
	glGetIntegerv(GL_MAJOR_VERSION, &glVersion[0]);
	glGetIntegerv(GL_MINOR_VERSION, &glVersion[1]);
	std::cout << "Running OpenGL " << glVersion[0] << "." << glVersion[1] << std::endl;

	if (glVersion[0] == -1 && glVersion[1] == -1)
	{
		SDL_GL_DeleteContext(context);
		SDL_DestroyWindow(window);

		std::cerr << "[OpenGL] Failed to create OpenGL context!" << std::endl;

		return 1;
	}

	// Main loop
	App app;
	if (!app.Init() || !app.InitCl())
	{
		SDL_GL_DeleteContext(context);
		SDL_DestroyWindow(window);
		std::cerr << "[app.Init] Error during app initialization!" << std::endl;
		return 1;
	}
	int w, h;
	SDL_GetWindowSize(window, &w, &h);
	app.Resize(w, h);

	bool quit = false;
	SDL_Event ev;
	while (!quit)
	{
		while (SDL_PollEvent(&ev))
		{
			switch (ev.type)
			{
			case SDL_QUIT:
				quit = true;
				break;
			case SDL_KEYDOWN:
				if (ev.key.keysym.sym == SDLK_ESCAPE)
					quit = true;
				app.KeyboardDown(ev.key);
				break;
			case SDL_KEYUP:
				app.KeyboardUp(ev.key);
				break;
			case SDL_MOUSEBUTTONDOWN:
				app.MouseDown(ev.button);
				break;
			case SDL_MOUSEBUTTONUP:
				app.MouseUp(ev.button);
				break;
			case SDL_MOUSEWHEEL:
				app.MouseWheel(ev.wheel);
				break;
			case SDL_MOUSEMOTION:
				app.MouseMove(ev.motion);
				break;
			case SDL_WINDOWEVENT:
				if (ev.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)
				{
					app.Resize(ev.window.data1, ev.window.data2);
				}
				break;
			}
		}

		app.Update();
		app.Render();

		SDL_GL_SwapWindow(window);
	}

	return 0;
}