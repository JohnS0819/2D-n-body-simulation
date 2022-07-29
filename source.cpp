#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <SFML/Graphics.hpp>
#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <complex>
#include <algorithm>
#include <atomic>
#include <thread>
#include <fstream>
#include <chrono>

cl::Program program;    // The program that will run on the device.    
cl::Context context;    // The context which holds the device.    
cl::Device device;      // The device where the kernel will run.
cl::Device getDefaultDevice() {
    
    /**
     * Search for all the OpenCL platforms available and check
     * if there are any.
     * */

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty()) {
        std::cerr << "No platforms found!" << std::endl;
        exit(1);
    }

    /**
     * Search for all the devices on the first platform and check if
     * there are any available.
     * */

    auto platform = platforms.front();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    if (devices.empty()) {
        std::cerr << "No devices found!" << std::endl;
        exit(1);
    }

    /**
     * Return the first device found.
     * */

    return devices.front();
}


void initializeDevice() {

    /**
     * Select the first available device.
     * */

    device = getDefaultDevice();

    /**
     * Read OpenCL kernel file as a string.
     * */

    std::ifstream kernel_file("Kernel.cl");
    std::string src(std::istreambuf_iterator<char>(kernel_file), (std::istreambuf_iterator<char>()));

    /**
     * Compile kernel program which will run on the device.
     * */

    cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));
    context = cl::Context(device);
    program = cl::Program(context, sources);

    auto err = program.build();
    if (err != CL_BUILD_SUCCESS) {
        std::cerr << "Error!\nBuild Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device)
            << "\nBuild Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        exit(1);
    }
}




struct particles {
    std::vector<cl_double2> position;
    std::vector<cl_double2> velocity;
    std::vector<cl_double> mass;
};


struct particle_data {
    std::complex<double> position;
    std::complex<double> velocity;
    double mass = 0;
    particle_data(std::complex<double> a, std::complex<double> b, double c) : position(a), velocity(b), mass(c) {};
    particle_data() {};
};

class render_body {
public:
    int size;
    std::vector<sf::Vertex> history;
    int index = 0;
    sf::CircleShape shape;
    render_body(particle_data initial, int path_length = 0) {
        shape.setRadius(10 * pow(initial.mass, 0.333333));
        shape.setOrigin(10 * pow(initial.mass, 0.333333), 10 * pow(initial.mass, 0.333333));
        shape.setPosition(initial.position.real(), initial.position.imag());
        shape.setPointCount(60);
        size = path_length + 1;
        for (int i = 0; i <= path_length; ++i) {
            history.push_back(sf::Vertex(sf::Vector2f(initial.position.real(), initial.position.imag())));
        }
    }
    render_body(cl_double2 position, cl_double mass, int path_length = 0) {
        shape.setRadius(10 * pow(mass, 0.333333));
        shape.setOrigin(10 * pow(mass, 0.333333), 10 * pow(mass, 0.333333));
        shape.setPosition(position.s0, position.s1);
        shape.setPointCount(60);
        size = path_length + 1;
        for (int i = 0; i <= path_length; ++i) {
            history.push_back(sf::Vertex(sf::Vector2f(position.s0, position.s1)));
        }
    }

    void update(cl_double2 new_vertex) {
        ++index;
        shape.setPosition(new_vertex.s0, new_vertex.s1);
        history[index % size] = (sf::Vertex(sf::Vector2f(new_vertex.s0, new_vertex.s1)));
    }

    void update(std::complex<double> new_vertex) {
        ++index;
        shape.setPosition(new_vertex.real(), new_vertex.imag());
        history[index % size] = (sf::Vertex(sf::Vector2f(new_vertex.real(), new_vertex.imag())));
    }
    void render_path(sf::RenderWindow& window) {
        window.draw(&history[0], history.size(), sf::Points);
    }
    void render(sf::RenderWindow& window) {
        window.draw(shape);
    }
};


class n_body_window {
    particles test;
    sf::Font font;
    std::vector<render_body> paths;
    particle_data temp_particle;
    int mouse_state = 0;

    cl::Buffer *mass_array = nullptr;
    cl::Buffer *position_array = nullptr;
    cl::Buffer *velocity_array = nullptr;
    cl::Buffer *position_buffer = nullptr;
    cl::Buffer *velocity_buffer = nullptr;
    int number_till_resizing_is_neccesary = 0;




    std::complex<double> force(particle_data& input, std::vector<particle_data>& data, int indice) {
        std::complex<double> output = { 0,0 };
        for (int i = 0; i < data.size(); ++i) {
            if (i == indice) {
                continue;
            }
            std::complex<double> direction = data[i].position - input.position;
            double distance = std::norm(direction);
            direction /= sqrt(distance);
            output += 0.2 * ((data[i].mass * direction) / (distance + 1));
        }
        return output;
    }
    //leapfrog integrator
    void leapfrog(std::vector<particle_data>& data, std::vector<particle_data>& buffer, double timestep) {
        for (int i = 0; i < data.size(); ++i) {
            std::complex<double> acceleration = force(data[i], data, i);
            buffer[i].position = data[i].position + timestep * (data[i].velocity + (acceleration * (timestep / 2)));
            buffer[i].velocity = data[i].velocity + (timestep / 2) * acceleration;
        }
        swap(data, buffer);
        for (int i = 0; i < data.size(); ++i) {
            std::complex<double> acceleration = force(data[i], data, i);
            data[i].velocity += (timestep / 2) * acceleration;
        }
    }

    void get_mouse_input(sf::RenderWindow& window, sf::View& view) {
        auto t = window.mapPixelToCoords(sf::Mouse::getPosition(window), view);
        switch (mouse_state) {
        case(0):
            temp_particle.position = std::complex<double>({ t.x,t.y });
            break;
        case(1):
            temp_particle.mass = pow(std::norm(temp_particle.position - std::complex<double>(t.x, t.y)) / 100, 1.5);
            break;
        case(2):
            temp_particle.velocity = (std::complex<double>(t.x, t.y) - temp_particle.position) / 1000.0;
            test.mass.push_back(temp_particle.mass);
            test.position.push_back({ temp_particle.position.real(),temp_particle.position.imag() });
            test.velocity.push_back({ temp_particle.velocity.real(),temp_particle.velocity.imag() });
            delete(mass_array);
            delete(position_array);
            delete(velocity_array);
            delete(position_buffer);
            delete(velocity_buffer);

            mass_array = new cl::Buffer(context, CL_MEM_READ_WRITE, (test.mass.size()) * sizeof(cl_double));
            position_array = new cl::Buffer(context, CL_MEM_READ_WRITE, (test.position.size()) * sizeof(cl_double2));
            velocity_array = new cl::Buffer(context, CL_MEM_READ_WRITE , (test.velocity.size()) * sizeof(cl_double2));
            position_buffer = new cl::Buffer(context, CL_MEM_HOST_NO_ACCESS, (test.velocity.size()) * sizeof(cl_double2));
            velocity_buffer = new cl::Buffer(context, CL_MEM_HOST_NO_ACCESS, (test.velocity.size()) * sizeof(cl_double2));
            cl::CommandQueue queue(context, device);
            queue.enqueueWriteBuffer(*position_array, CL_FALSE, 0, test.position.size() * sizeof(cl_double2),test.position.data());
            queue.enqueueWriteBuffer(*velocity_array, CL_FALSE, 0, test.position.size() * sizeof(cl_double2), test.velocity.data());
            queue.enqueueWriteBuffer(*mass_array, CL_FALSE, 0, test.position.size() * sizeof(cl_double), test.mass.data());
            
            paths.push_back({ temp_particle,10000 });
        }

        mouse_state = (mouse_state + 1) % 3;
    }

    void render(sf::RenderWindow& window, sf::View& view) {
        auto coordinates = window.mapPixelToCoords(sf::Mouse::getPosition(window), view);
        sf::Text text;
        text.setFont(font);
        text.setPosition((-0.5f * view.getSize()) + view.getCenter());
        text.setScale((0.0005555555f) * view.getSize().x, (0.0005555555f) * view.getSize().x);
        switch (mouse_state) {
        case(0):
            text.setString(std::to_string(coordinates.x) + "\t" + std::to_string(coordinates.y) + "\t" + std::to_string((int)test.position.size()));
            break;
        case(1):
        {
            sf::CircleShape render_shape;
            render_shape.setPosition(sf::Vector2f(temp_particle.position.real(), temp_particle.position.imag()));
            std::complex<double> p2 = { window.mapPixelToCoords(sf::Mouse::getPosition(window),view).x,window.mapPixelToCoords(sf::Mouse::getPosition(window),view).y };
            render_shape.setRadius(sqrt(std::norm(temp_particle.position - p2)));
            render_shape.setOrigin(sqrt(std::norm(temp_particle.position - p2)), sqrt(std::norm(temp_particle.position - p2)));
            render_shape.setPointCount(60);
            text.setString(std::to_string(pow(std::norm(temp_particle.position - p2) / 100, 1.5)));
            window.draw(render_shape);
        }
        break;
        case(2):
            sf::Vertex line[2];
            line[0].position = sf::Vector2f(temp_particle.position.real(), temp_particle.position.imag());
            line[1].position = window.mapPixelToCoords(sf::Mouse::getPosition(window), view);
            auto velocity = line[1].position - line[0].position;
            text.setString(std::to_string(velocity.x / 1000.0) + "\t" + std::to_string(velocity.y / 1000.0));
            window.draw(line, 2, sf::Lines);
        }
        window.draw(text);

        for (int i = 0; i < paths.size(); ++i) {
            paths[i].render(window);
            paths[i].render_path(window);
        }
    }

    
    //simulate n-body interactions with given iteration number and timestep
    void update(int iterations, double timestep) {
        auto t3 = std::chrono::high_resolution_clock::now();
        int N = test.position.size();

        cl::Kernel leapfrogpart1(program, "leapfrog1");
        cl::Kernel leapfrogpart2(program, "leapfrog2");
        cl::Kernel leapfrogpart3(program, "leapfrog_part_2_1");
        cl::Kernel leapfrogpart4(program, "leapfrog_part_2_2");
        cl::Kernel basic(program, "euler");



        leapfrogpart1.setArg(0, *position_array);
        leapfrogpart1.setArg(1, *velocity_array);
        leapfrogpart1.setArg(2, *mass_array);
        leapfrogpart1.setArg(3, *position_buffer);
        leapfrogpart1.setArg(4, *velocity_buffer);
        leapfrogpart1.setArg(5, timestep);
        leapfrogpart1.setArg(6, N);

        leapfrogpart2.setArg(0, *position_buffer);
        leapfrogpart2.setArg(1, *velocity_buffer);
        leapfrogpart2.setArg(2, *mass_array);
        leapfrogpart2.setArg(3, timestep);
        leapfrogpart2.setArg(4, N);

        leapfrogpart3.setArg(3, *position_array);
        leapfrogpart3.setArg(4, *velocity_array);
        leapfrogpart3.setArg(2, *mass_array);
        leapfrogpart3.setArg(0, *position_buffer);
        leapfrogpart3.setArg(1, *velocity_buffer);
        leapfrogpart3.setArg(5, timestep);
        leapfrogpart3.setArg(6, N);

        leapfrogpart4.setArg(0, *position_array);
        leapfrogpart4.setArg(1, *velocity_array);
        leapfrogpart4.setArg(2, *mass_array);
        leapfrogpart4.setArg(3, timestep);
        leapfrogpart4.setArg(4, N);

        auto t4 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> fp = t4 - t3;
        std::cout << fp.count() << "microseconds\t";




        cl::CommandQueue queue(context, device);

        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {

            queue.enqueueNDRangeKernel(leapfrogpart1, cl::NullRange, cl::NDRange(N));
            queue.enqueueNDRangeKernel(leapfrogpart2, cl::NullRange, cl::NDRange(N));
            queue.enqueueNDRangeKernel(leapfrogpart3, cl::NullRange, cl::NDRange(N));
            queue.enqueueNDRangeKernel(leapfrogpart4, cl::NullRange, cl::NDRange(N));
            /*
            basic.setArg(0, position_array);
            basic.setArg(1, velocity_array);
            basic.setArg(3, position_buffer);
            basic.setArg(4, velocity_buffer);
            basic.setArg(2, mass_array);
            basic.setArg(5, timestep);
            basic.setArg(6, N);
            queue.enqueueNDRangeKernel(basic, cl::NullRange, cl::NDRange(N));
            basic.setArg(0, position_buffer);
            basic.setArg(1, velocity_buffer);
            basic.setArg(3, position_array);
            basic.setArg(4, velocity_array);
            basic.setArg(2, mass_array);
            basic.setArg(5, timestep);
            basic.setArg(6, N);
            queue.enqueueNDRangeKernel(basic, cl::NullRange, cl::NDRange(N));
            */

        }
        
        queue.enqueueReadBuffer(*position_array, CL_TRUE, 0, N * sizeof(cl_double2), test.position.data());
        queue.enqueueReadBuffer(*velocity_array, CL_TRUE, 0, N * sizeof(cl_double2), test.velocity.data());
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> fp_ms = t2 - t1;
        std::cout << fp_ms.count() << "microseconds" << std::endl;

        for (int i = 0; i < N; ++i) {
            paths[i].update(test.position[i]);
        }


        /*
        for (int i = 0; i < iterations; ++i) {
            leapfrog(data, buffer, timestep);

        }
        for (int i = 0; i < data.size(); ++i) {
            paths[i].update(data[i].position);
        }
        */
    }


public:
    n_body_window(std::vector<particle_data> a, std::vector<render_body> b) {
        paths = b;
        font.loadFromFile("times.ttf");


        for (int i = 0; i < 1000; ++i) {
            auto radius = 3000 + (rand() % 1000);
            auto angle = rand() % 1000000;

            particle_data temp;
            temp.mass = 1;
            test.mass.push_back(1);
            test.position.push_back( cl_double2({radius * cos(angle),radius * sin(angle) }));
            test.velocity.push_back(cl_double2({45 * (1.00/sqrt(radius)) *  sin(-1 * angle),  45 * (1.00 / sqrt(radius)) * cos(angle) }));
            temp.position = { test.position[i].s0,test.position[i].s1 };
            temp.velocity = { test.velocity[i].s0,test.velocity[i].s1 };
            //data.push_back(temp);
            paths.push_back({ temp, 100 });
        }
        particle_data temp;
        temp.mass = 100000;
        temp.position = { 0,0 };
        paths.push_back({ temp,0 });
        test.position.push_back({ 0,0 });
        test.velocity.push_back({ 0,0 });
        test.mass.push_back(10000);



        mass_array = new cl::Buffer(context, CL_MEM_COPY_HOST_PTR, test.mass.size() * sizeof(cl_double), test.mass.data());
        position_array = new cl::Buffer(context, CL_MEM_COPY_HOST_PTR, test.position.size() * sizeof(cl_double2), test.position.data());
        velocity_array = new cl::Buffer(context, CL_MEM_COPY_HOST_PTR, test.velocity.size() * sizeof(cl_double2), test.velocity.data());
        position_buffer = new cl::Buffer(context, CL_MEM_HOST_NO_ACCESS, test.velocity.size() * sizeof(cl_double2));
        velocity_buffer = new cl::Buffer(context, CL_MEM_HOST_NO_ACCESS, test.velocity.size() * sizeof(cl_double2));



    }

    void operator()(std::atomic<int>& state, std::atomic<bool>& window_is_open) {
        double timestep = 0.5;
        float view_scale = 1.0;
        sf::RenderWindow window(sf::VideoMode(1800, 1100), "n-body simulation");
        sf::Vector2f last_size = { 1800,1100 };
        sf::View view(sf::FloatRect(0, 0, 1800, 1100));
        window.setVerticalSyncEnabled(true);
        bool mouse_wheel_pressed = false;
        sf::Vector2f pan(0, 0);
        int iterations = 1;
        //main render loop
        while (window.isOpen()) {
            sf::Event event;
            //event queue loop
            while (window.pollEvent(event)) {
                switch (event.type) {
                case(sf::Event::Closed):
                    window.close();
                    window_is_open = false;
                    return;
                    break;

                case(sf::Event::MouseButtonReleased):
                    mouse_wheel_pressed = false;
                    break;

                case(sf::Event::MouseWheelScrolled):
                    view_scale *= pow(0.9, event.mouseWheelScroll.delta);
                    view.zoom(pow(0.9, event.mouseWheelScroll.delta));
                    break;

                case(sf::Event::Resized): {
                    auto current = sf::Vector2f(window.getSize());
                    auto transform = view.getSize();
                    transform.x *= (current.x / last_size.x);
                    transform.y *= (current.y / last_size.y);
                    last_size = current;
                    view.setSize(transform);
                    break;
                }
                case(sf::Event::MouseButtonPressed):
                    if (event.mouseButton.button != sf::Mouse::Left) {
                        break;
                    }
                    get_mouse_input(window, view);

                case(sf::Event::KeyPressed):
                    if (event.key.code == sf::Keyboard::Add) {
                        iterations = iterations == 0 ? 1 : iterations == 1048576 ? iterations : iterations * 2;
                    }
                    if (event.key.code == sf::Keyboard::Subtract) {
                        iterations /= 2;
                    }
                    break;
                }
                if (sf::Mouse::isButtonPressed(sf::Mouse::Middle)) {
                    mouse_wheel_pressed ? pan = pan : pan = sf::Vector2f(sf::Mouse::getPosition(window));
                    mouse_wheel_pressed = true;

                    view.move(view_scale * (sf::Vector2f(sf::Mouse::getPosition(window)) - pan));
                    pan = sf::Vector2f(sf::Mouse::getPosition(window));
                }
            }

            switch (state) {
            case(2):
                update(iterations, timestep);
                break;
            case(0):
                window.close();
                window_is_open = false;
                return;
            }
            window.clear(sf::Color::Black);
            window.setView(view);
            render(window, view);
            window.display();
        }
    }
};


int main()
{
    sf::Font font;
    if (!font.loadFromFile("times.ttf")) {
        std::cout << "failed to load font";
        return 0;
    }
    initializeDevice();
    //state variables
    std::atomic<int> t1 = 2;
    std::atomic<bool> t2 = false;

    std::vector<std::string> commands = { "help","pause","continue","clear","new" };
    while (true) {
        std::string temp;
        std::cin >> temp;
        if (std::find(commands.begin(), commands.end(), temp) == commands.end()) {
            std::cout << "unrecognized command press help for list of commands" << std::endl;
            continue;
        }
        int index = std::find(commands.begin(), commands.end(), temp) - commands.begin();
        switch (index) {
        case(0):
            std::cout << "help, pause, clear, new, continue" << std::endl;
            break;
        case(1):
            t2 ? t1 = 1 : t1 = 2;
            break;
        case(2):
            t1 = 2;
            break;
        case(3):
            t1 = 0;
            break;
        case(4):
            if (t2) {
                std::cout << "theres already an active window" << std::endl;
                continue;
            }
            t2 = true;
            t1 = 2;
            std::vector<particle_data> data;
            std::vector<render_body> paths;
            n_body_window window(data, paths);
            std::thread thread(window, std::ref(t1), std::ref(t2));
            thread.detach();
        }


    }
    return 0;
}
