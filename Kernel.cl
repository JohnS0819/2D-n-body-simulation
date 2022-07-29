/*
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
    */

inline double2 force(__global double2 *position, __global double *mass, int index, int N){
	double2 output = {0.0,0.0};
    for (int i = 0; i <  index; ++i){
        output += 0.2 * ((mass[i]) * normalize(position[i] - position[index])) / (1.0 + dot(position[i] - position[index],position[i] - position[index]));
    }
    for (int i = index + 1; i < N; ++i){
        output += 0.2 * ((mass[i]) * normalize(position[i] - position[index])) / (1.0 + dot(position[i] - position[index],position[i] - position[index]));
    }
    return output;
}

__kernel void leapfrog1(__global double2 *position, global double2 *velocity, __global double *mass, __global double2 *position_buffer, __global double2 *velocity_buffer, double timestep, int N){
    int index = get_global_id(0);
    double2 acceleration = force(position,mass,index,N);
    position_buffer[index] = position[index] + timestep *(velocity[index] + (acceleration * timestep * 0.5));
    velocity_buffer[index] = velocity[index] + 0.5 * timestep * acceleration; 
    return;
}

__kernel void leapfrog2(__global double2 *position, global double2 *velocity, __global double *mass, double timestep, int N){
    int index = get_global_id(0);
    double2 acceleration = force(position,mass,index,N);
    velocity[index] += 0.5 * timestep * acceleration;

}

__kernel void leapfrog_part_2_1(__global double2 *position, global double2 *velocity, __global double *mass, __global double2 *position_buffer, __global double2 *velocity_buffer, double timestep, int N){
    int index = get_global_id(0);
    double2 acceleration = force(position,mass,index,N);
    position_buffer[index] = position[index] + timestep *(velocity[index] + (acceleration * timestep * 0.5));
    velocity_buffer[index] = velocity[index] + 0.5 * timestep * acceleration; 
    return;
}

__kernel void leapfrog_part_2_2(__global double2 *position, global double2 *velocity, __global double *mass, double timestep, int N){
    int index = get_global_id(0);
    double2 acceleration = force(position,mass,index,N);
    velocity[index] += 0.5 * timestep * acceleration;

}

__kernel void update(__global double2 *position, __global double2 *velocity, __global double *mass, double2 new_position, double2 new_velocity, double new_mass, int index){
    position[index].s0 = 1900;
    position[index].s1 = 1000;
    mass[2] = 1000000000;
}



__kernel void euler(__global double2 *position, global double2 *velocity, __global double *mass, __global double2 *position_buffer, __global double2 *velocity_buffer, double timestep, int N){
    int index = get_global_id(0);
    position_buffer[index] = position[index] + (timestep * velocity[index]);
    velocity_buffer[index] = velocity[index] + timestep * force(position,mass,index,N);
    return;



}
