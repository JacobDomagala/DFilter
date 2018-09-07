#include<GL\glew.h>
#include<SOIL.h>
#include<SDL.h>
#include<iostream>
#include"cuda_runtime.h"
#include"device_launch_parameters.h"

#define myCeil(num1,num2) num1 % num2 == 0 ? num1/num2 : 1 + num1/num2 
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) system("pause"); exit(code);
	}
}

#pragma region DEBUG_SHADERS
bool CheckStatus(GLuint object, PFNGLGETSHADERIVPROC objectPropertyGetterFunc,
	PFNGLGETSHADERINFOLOGPROC getInfoLogFunc, GLenum statusType)
{
	GLint Status;
	objectPropertyGetterFunc(object, statusType, &Status);
	if (Status != GL_TRUE)
	{
		GLint infoLogLength;
		objectPropertyGetterFunc(object, GL_INFO_LOG_LENGTH, &infoLogLength);
		GLchar * buffer = new GLchar[infoLogLength];

		GLsizei bufferSize;
		getInfoLogFunc(object, infoLogLength, &bufferSize, buffer);
		std::cout << buffer << std::endl;

		delete[] buffer;
		return false;
	}
	return true;
}
bool CheckShaderStatus(GLuint shader)
{
	return CheckStatus(shader, glGetShaderiv, glGetShaderInfoLog, GL_COMPILE_STATUS);
}
bool CheckProgramStatus(GLuint program)
{
	return CheckStatus(program, glGetProgramiv, glGetProgramInfoLog, GL_LINK_STATUS);
}
#pragma endregion

int SDLCALL myEventFilter(void *userdata, SDL_Event *e)
{
		if (e->type == SDL_WINDOWEVENT)
		{
			if (e->window.event == SDL_WINDOWEVENT_RESIZED)
			{
				int width = e->window.data1;
				int height = e->window.data2;
				glViewport(0, 0, width, height);
				glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
				SDL_GL_SwapWindow((SDL_Window*)userdata);
			}
		}
		return 1; // return 1 so all events are added to queue
}
#pragma region GPU
__global__ void SobelOperator(const uchar3* inputChannel, uchar3* outputChannel, int numRows, int numCols)
{
	int threadX = threadIdx.x + blockDim.x * blockIdx.x;
	int threadY = threadIdx.y + blockDim.y * blockIdx.y;
	int id = threadX + threadY * numCols;

	if (threadX >= numCols || threadY >= numRows)
		return;

	char xDirection[] = {
		-1, 0, 1,
		-2, 0, 2,
		-1, 0, 1,
	};
	char yDirection[] = {
		-1, -2, -1,
		 0,  0,  0,
		 1,  2,  1,
	};
	char filterWidth = 3;

	float resultX = 0.0f;
	float resultY = 0.0f;

	for (int filterY = -filterWidth / 2; filterY <= filterWidth / 2; ++filterY) {
		for (int filterX = -filterWidth / 2; filterX <= filterWidth / 2; ++filterX)
		{
			int imageY = min(max(threadY + filterY, 0), (numRows - 1));
			int imageX = min(max(threadX + filterX, 0), (numCols - 1));

			float imageValue = inputChannel[imageY * numCols + imageX].x;
			float filterValueX = xDirection[(filterY + filterWidth / 2) * filterWidth + filterX + filterWidth / 2];
			float filterValueY = yDirection[(filterY + filterWidth / 2) * filterWidth + filterX + filterWidth / 2];

			resultX += imageValue * filterValueX;
			resultY += imageValue * filterValueY;
		}
	}
	
	float resultXSquared = resultX * resultX;
	float resultYSquared = resultY * resultY;

	float finalResult = sqrt(resultXSquared + resultYSquared);

	outputChannel[id].x = finalResult;
	outputChannel[id].y = finalResult;
	outputChannel[id].z = finalResult;
}
__global__ void GaussianBlur(const unsigned char* const inputChannel, unsigned char* const outputChannel, int numRows, int numCols, const float* const filter, const int filterWidth)
{
	int threadX = threadIdx.x + blockDim.x * blockIdx.x;
	int threadY = threadIdx.y + blockDim.y * blockIdx.y;
	int id = threadX + threadY * numCols;

	if (threadX >= numCols || threadY >= numRows)
		return;

	float result = 0.0f;
	float normalize = 0.0f;
	for (int filterY = -filterWidth / 2; filterY <= filterWidth / 2; ++filterY) {
		for (int filterX = -filterWidth / 2; filterX <= filterWidth / 2; ++filterX)
		{
			//clamp to boundary of the image
			int imageY = min(max(threadY + filterY, 0), (numRows - 1));
			int imageX = min(max(threadX + filterX, 0), (numCols - 1));

			float imageValue = inputChannel[imageY * numCols + imageX];
			float filterValue = filter[(filterY + filterWidth / 2) * filterWidth + filterX + filterWidth / 2];

			// zsumuj kolejny iloczyn warto�ci filtra z warto�ci� piksela oraz zsumuj warto�� filtra
			result += imageValue * filterValue;
			normalize += filterValue;
		}
	}

	result = result / normalize;

	outputChannel[id] = result;
}
__global__ void RGBAToGrayscale(const uchar3* const rgbaImage, uchar3* const greyImage, int numRows, int numCols)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int id = idx + idy*numCols;
	if (idx >= numCols || idy >= numRows)
		return;
	float tmp = rgbaImage[id].x * .299f + rgbaImage[id].y * .587f + rgbaImage[id].z * .114f;

	greyImage[id].x = tmp;
	greyImage[id].y = tmp;
	greyImage[id].z = tmp;
}

__global__ void SeparateChannels(uchar3* d_input, unsigned char* d_red, unsigned char* d_green, 
	unsigned char* d_blue, unsigned int numCols, unsigned int numRows)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	int id = idx + idy*numCols;
	
	if (idx >= numCols || idy >= numRows)
		return;

	unsigned char red, green, blue;
	red = d_input[id].x;
	green = d_input[id].y;
	blue = d_input[id].z;

	d_red[id] = red;
	d_green[id] = green;
	d_blue[id] = blue;
}
__global__ void CombineChannels(uchar3* d_output, unsigned char* d_red, unsigned char* d_green, unsigned char* d_blue, unsigned int numCols, unsigned int numRows)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	int id = idx + idy*numCols;

	if (idx >= numCols || idy >= numRows)
		return;
	
	unsigned char red, green, blue;
	red = d_red[id];
	green = d_green[id];
	blue = d_blue[id];

	d_output[id].x = red;
	d_output[id].y = green;
	d_output[id].z = blue;
}
#pragma endregion

#undef main
int main(int argc, char** argv)
{
#pragma region SHADERS

	const GLchar* vertexShader ={
	"#version 430 core\n"
	" \n"
	" out vec2 fTexCoord;\n"
	" void main(){\n"
	" const vec4 positions[4] = vec4[4]( vec4(-1.0f, 1.0f, 0.0f, 1.0f),\n"
	" vec4(-1.0f, -1.0f, 0.0f, 1.0f),\n "
	" vec4(1.0f, -1.0f, 0.0f, 1.0f),\n "
	" vec4(1.0f, 1.0f, 0.0f, 1.0f));\n "
	" const vec2 texCoords[4] = vec2[4]( vec2(0.0f, 0.0f),\n"
	" vec2(0.0f, 1.0f),\n "
	" vec2(1.0f, 1.0f),\n "
	" vec2(1.0f, 0.0f));\n "
	" \n"
	" gl_Position = positions[gl_VertexID];\n"
	" fTexCoord = texCoords[gl_VertexID];\n"
	"}"
	};

	const GLchar* fragmentShader ={
	"#version 430 core\n"
	" in vec2 fTexCoord;\n"
	" uniform sampler2D texture;\n"
	" out vec4 color;\n"
	" void main(){\n"
	" color = texture2D(texture, fTexCoord);\n"
	"}"
	};
#pragma endregion
#pragma region SETUP

	SDL_Init(SDL_INIT_VIDEO);

	SDL_GLContext context;
	SDL_Window* windowHandle;
	SDL_Event e;
	
	SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_BUFFER_SIZE, 24);

	windowHandle = SDL_CreateWindow("Praca dyplomowa", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 800, 600, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
	SDL_AddEventWatch(myEventFilter, windowHandle);

	context = SDL_GL_CreateContext(windowHandle);

	GLenum status = glewInit();
	if (status != GLEW_OK)
	{
		std::cerr << "Wystapil problem z inicjalizacja Opengl(glew)!" << std::endl;
		system("pause");
	}

	GLuint vertexShaderID = glCreateShader(GL_VERTEX_SHADER);
	GLuint fragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);
	
	glShaderSource(vertexShaderID, 1, &vertexShader, 0);
	glShaderSource(fragmentShaderID, 1, &fragmentShader, 0);
	glCompileShader(vertexShaderID);
	glCompileShader(fragmentShaderID);

	if (!CheckShaderStatus(vertexShaderID) || !CheckShaderStatus(fragmentShaderID))
	{
		system("pause");
	}

	GLuint programID = glCreateProgram();

	glAttachShader(programID, vertexShaderID);
	glAttachShader(programID, fragmentShaderID);
	glLinkProgram(programID);

	if (!CheckProgramStatus(programID))
	{
		system("pause");
	}

	glUseProgram(programID);
	
	int width, height;

	unsigned char* data = SOIL_load_image("flower.jpg", &width, &height, 0, SOIL_LOAD_RGB);
	if (data == NULL)
		printf("Unable to load image!");

	GLuint textureID;

	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
	glUniform1i(glGetUniformLocation(programID, "texture"), 0);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

#pragma endregion

	const int blurKernelWidth = 9;
	const float blurKernelSigma = 2.0f;

	float* h_filter = new float[blurKernelWidth * blurKernelWidth];

	for (int y = -blurKernelWidth / 2; y <= blurKernelWidth / 2; ++y) 
	{
		for (int x = -blurKernelWidth / 2; x <= blurKernelWidth / 2; ++x)
		{
			float filterValue = expf(-(float)(x * x + y * y) / (2.f * blurKernelSigma * blurKernelSigma));
			int index = (y + blurKernelWidth / 2) * blurKernelWidth + x + blurKernelWidth / 2;
			h_filter[index] = filterValue;
		}
	}


	size_t kernelWidth = blurKernelWidth;


	uchar3* d_data;
	uchar3* d_result;
	float* d_gaussianKernel;
	unsigned char* d_red;
	unsigned char* d_redBlurred;
	unsigned char* d_green;
	unsigned char* d_greenBlurred;
	unsigned char* d_blue;
	unsigned char* d_blueBlurred;

	size_t kernelSize = kernelWidth * kernelWidth * sizeof(float);
	size_t numPixels = width * height;
	size_t channelSize = numPixels * sizeof(unsigned char);
	size_t picSize = channelSize * 3;
	
	gpuErrchk(cudaMalloc(&d_gaussianKernel, kernelSize));
	gpuErrchk(cudaMalloc(&d_data, picSize));
	gpuErrchk(cudaMalloc(&d_result, picSize));
	gpuErrchk(cudaMalloc(&d_red, channelSize));
	gpuErrchk(cudaMalloc(&d_green, channelSize));
	gpuErrchk(cudaMalloc(&d_blue, channelSize));
	gpuErrchk(cudaMalloc(&d_redBlurred, channelSize));
	gpuErrchk(cudaMalloc(&d_greenBlurred, channelSize));
	gpuErrchk(cudaMalloc(&d_blueBlurred, channelSize));
	gpuErrchk(cudaMemcpy(d_data, data, picSize, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_gaussianKernel, h_filter, kernelSize, cudaMemcpyHostToDevice));

	dim3 blockSize(32,32);
	unsigned int gridX = myCeil(width, blockSize.x);
	unsigned int gridY = myCeil(height, blockSize.x);
	dim3 gridSize(gridX, gridY);
	
	bool isOpen = true;
	while (isOpen)
	{
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		while (SDL_PollEvent(&e))
		{
			if (e.type == SDL_QUIT)
			{
				isOpen = false;
			}
			if (e.type == SDL_KEYDOWN)
			{
				//GRAYSCALE 
				if (e.key.keysym.scancode == SDL_SCANCODE_D)
				{
					RGBAToGrayscale <<< gridSize, blockSize >>>(d_data, d_result, height, width);
					cudaMemcpy(data, d_result, picSize, cudaMemcpyDeviceToHost);
					glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
				}
				//SOBEL EDGE DETECTION
				if (e.key.keysym.scancode == SDL_SCANCODE_W)
				{
					RGBAToGrayscale <<< gridSize, blockSize >>>(d_data, d_result, height, width);
					cudaDeviceSynchronize();
					SobelOperator <<<gridSize, blockSize >>>(d_result, d_data, height, width);
					cudaMemcpy(data, d_data, picSize, cudaMemcpyDeviceToHost);
					glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
				}
				//GAUSSIAN BLUR
				if (e.key.keysym.scancode == SDL_SCANCODE_S)
				{
					SeparateChannels <<< gridSize, blockSize >>>(d_data, d_red, d_green, d_blue, width, height);
					gpuErrchk(cudaPeekAtLastError());
					gpuErrchk(cudaDeviceSynchronize());
					GaussianBlur <<< gridSize, blockSize >>>(d_red, d_redBlurred, height, width, d_gaussianKernel, kernelWidth);
					gpuErrchk(cudaPeekAtLastError());
					gpuErrchk(cudaDeviceSynchronize());
					GaussianBlur <<< gridSize, blockSize >>>(d_green, d_greenBlurred, height, width, d_gaussianKernel, kernelWidth);
					gpuErrchk(cudaPeekAtLastError());
					gpuErrchk(cudaDeviceSynchronize());
					GaussianBlur <<< gridSize, blockSize >>>(d_blue, d_blueBlurred, height, width, d_gaussianKernel, kernelWidth);
					gpuErrchk(cudaPeekAtLastError());
					gpuErrchk(cudaDeviceSynchronize());
					CombineChannels <<< gridSize, blockSize >>>(d_data, d_redBlurred, d_greenBlurred, d_blueBlurred, width, height);
					gpuErrchk(cudaPeekAtLastError());
					gpuErrchk(cudaDeviceSynchronize());
					gpuErrchk(cudaMemcpy(data, d_data, picSize, cudaMemcpyDeviceToHost));
					glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
				}
				if (e.key.keysym.scancode == SDL_SCANCODE_R)
				{
					data = SOIL_load_image("flower.jpg", &width, &height, 0, SOIL_LOAD_RGB);
					glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
					cudaMemcpy(d_data, data, picSize, cudaMemcpyHostToDevice);
				}
			}
		}
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
		SDL_GL_SwapWindow(windowHandle);
	}
	

	// clean up
	SOIL_free_image_data(data);
	glDeleteShader(vertexShaderID);
	glDeleteShader(fragmentShaderID);
	glDeleteProgram(programID);
	SDL_GL_DeleteContext(context);
	SDL_DestroyWindow(windowHandle);
	SDL_Quit();
}