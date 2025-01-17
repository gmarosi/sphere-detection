#include "App.h"

App::App()
{
	camera.SetView(glm::vec3(5, 5, 5), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
	pointCloud = nullptr;
}

bool App::Init()
{
	glClearColor(0.125f, 0.25f, 0.5f, 1.0f);

	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);

	pointCloud = new PointCloud();

	// Constructing the data structure for passing buffer names to PointCloud
	MemoryNames memNames;
	char buf1[] = "sync_mem";
	memNames.first = std::wstring(buf1, buf1 + strlen(buf1));

	std::vector<std::pair<std::wstring, std::wstring>> snd;
	char buf2[] = "shm_1";
	char buf3[] = "shm_2";
	snd.push_back(std::make_pair(std::wstring(buf2, buf2 + strlen(buf2)), std::wstring(buf3, buf3 + strlen(buf3))));
	memNames.second = snd;

	pointCloud->Init(memNames);

	axisProgram.AttachShaders({
		{GL_VERTEX_SHADER,	 "axis.vert"},
		{GL_FRAGMENT_SHADER, "axis.frag"}
		});

	axisProgram.LinkProgram();

	camera.SetProj(glm::radians(60.0f), 640.0f / 480.0f, 0.01f, 1000.0f);

	return true;
}

bool App::InitCl()
{
	try
	{
		cl::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);

		bool create_context_success = false;
		for (auto platform : platforms)
		{
			cl_context_properties props[] =
			{
				CL_CONTEXT_PLATFORM,	(cl_context_properties)(platform)(),
				CL_GL_CONTEXT_KHR,		(cl_context_properties) wglGetCurrentContext(),
				CL_WGL_HDC_KHR,			(cl_context_properties)wglGetCurrentDC(),
				0
			};

			try
			{
				context = cl::Context(CL_DEVICE_TYPE_GPU, props);
				create_context_success = true;
				break;
			}
			catch (cl::Error error) {}
		}

		if (!create_context_success)
			throw cl::Error(CL_INVALID_CONTEXT, "Failed to create CL/GL shared context");

		cl::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
		commandQueue = cl::CommandQueue(context, devices[0]);

		return pointCloud->InitCl(context, devices);
	}
	catch (cl::Error error)
	{
		std::cout << error.what() << std::endl;
		return false;
	}
}

void App::Clean()
{

}

void App::Update()
{
	static Uint32 last_time = SDL_GetTicks();
	float delta_time = (SDL_GetTicks() - last_time) / 1000.0f;

	camera.Update(delta_time);
	pointCloud->Update();

	last_time = SDL_GetTicks();
}

void App::Render()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glm::mat4 viewProj = camera.GetViewProj();

	axisProgram.Use();
	axisProgram.SetUniform("MVP", viewProj * glm::mat4(1.0f) * glm::translate<float>(glm::vec3(0, 3, 0)));
	glDrawArrays(GL_LINES, 0, 6);
	axisProgram.Unuse();

	pointCloud->Render(viewProj);
}

void App::KeyboardDown(SDL_KeyboardEvent& key)
{
	camera.KeyboardDown(key);
}

void App::KeyboardUp(SDL_KeyboardEvent& key)
{
	camera.KeyboardUp(key);
}

void App::MouseMove(SDL_MouseMotionEvent& mouse)
{
	camera.MouseMove(mouse);
}

void App::MouseDown(SDL_MouseButtonEvent& mouse)
{
}

void App::MouseUp(SDL_MouseButtonEvent& mouse)
{
}

void App::MouseWheel(SDL_MouseWheelEvent& wheel)
{
}

// a két paraméterbe az új ablakméret szélessége (_w) és magassága (_h) található
void App::Resize(int _w, int _h)
{
	glViewport(0, 0, _w, _h);

	camera.Resize(_w, _h);
}