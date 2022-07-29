# 2D-n-body-simulation


Hardware accelerated version of original project. Performs the leapfrog integration using an OpenCL kernel rather than the previous single threaded implementation. This provides the benefit of highly parallelized force compuatation but at the expense of hardware latency and overhead. This makes it a good fit for larger numbers of particles, however the overhead restricts how many substeps can be calculated in each frame. This makes it far less suitable for lengthy simulations involving lower numbers of particles (n < ~800). 
