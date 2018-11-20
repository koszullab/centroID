
__author__ = "hervemn"
import numpy as np
import pycuda.tools
from pycuda import characterize
import pycuda.driver as cuda
import pycuda.compiler
from pycuda import gpuarray as ga

cuda.init()
num_gpu = cuda.Device.count()
for i in range(0, num_gpu):
    tmp_dev = cuda.Device(i)
    print("device_id = ", i, tmp_dev.name())
id_gpu = input("Select GPU: ")
id_gpu = int(id_gpu)
curr_gpu = cuda.Device(id_gpu)
print("you have selected ", curr_gpu.name())
kernels_cuda_src = """
    #include <curand_kernel.h>

    extern "C"
    {

        __global__ void init_rng(int nthreads, curandState *s, unsigned long long seed, unsigned long long offset)
        {
                int id = blockIdx.x*blockDim.x + threadIdx.x;

                if (id >= nthreads)
                        return;
                curand_init(seed, id, offset, &s[id]);
        }


        __global__ void gen_rand_mat(int* initArray, float *randArray, curandState *state, int n_rng, int width,
                                     float fact_sub)
        {
            int r0 = threadIdx.x + blockDim.x * blockIdx.x;
            int r1 = threadIdx.y + blockDim.y * blockIdx.y;
            int coord = r0 * width + r1;
            int id_rng = coord % n_rng ;
            if ((r0<width) && (r1< width)) {
                float mean = initArray[coord] * fact_sub;
                randArray[coord] = curand_poisson(&state[id_rng], mean);
            }
        }

        __global__ void copy_mat(int* initArray, float *randArray, int width, float fact_sub)
        {
            int r0 = threadIdx.x + blockDim.x * blockIdx.x;
            int r1 = threadIdx.y + blockDim.y * blockIdx.y;
            int coord = r0 * width + r1;
            if ((r0<width) && (r1< width)) {
                float mean = initArray[coord] * fact_sub;
                randArray[coord] = mean;
            }
        }

        __global__ void sum_along_axis(float* input, float* vect_sum, int width , int axis)
        {
            int r0 = threadIdx.x + blockDim.x * blockIdx.x;
            int r1 = threadIdx.y + blockDim.y * blockIdx.y;
            int coord = r0 * width + r1;
            if ((r0<width) && (r1< width)) {
                float val = input[coord];
                int id = (axis == 0) * r0 + (axis == 1) * r1;
                val = atomicAdd( &(vect_sum[id]), (float) val);
            }
        }

        __global__ void norm_along_axis(float* input, float* vect_sum, int width, int axis)
        {
            int r0 = threadIdx.x + blockDim.x * blockIdx.x;
            int r1 = threadIdx.y + blockDim.y * blockIdx.y;
            int coord = r0 * width + r1;
            if ((r0<width) && (r1< width)) {
                float val = input[coord];
                int id = (axis == 0) * r0 + (axis == 1) * r1;
                input[coord] = val / vect_sum[id];
            }
        }
        __global__ void init_vect_sum(float* vect_sum, int width)
            {
                int r0 = threadIdx.x + blockDim.x * blockIdx.x;
                if (r0 < width){
                    vect_sum[r0] = 0;
                }
            }

    } // extern "C"
"""


def get_rng_states(size_output, seed=1):
    init_rng_src = """
        #include <curand_kernel.h>

        extern "C"
        {

            __global__ void init_rng(int nthreads, curandState *s, unsigned long long seed, unsigned long long offset)
            {
                    int id = blockIdx.x*blockDim.x + threadIdx.x;

                    if (id >= nthreads)
                            return;
                    curand_init(seed, id, offset, &s[id]);
            }

            __global__ void make_rand(int nthreads, curandState *state, int *randArray)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                int id_rng = blockIdx.x;
                double mean = 10;
                if (idx<= nthreads){
                randArray[idx] = curand_poisson(&state[id_rng], mean);
                //randArray[idx] = curand_uniform(&state[idx]);
                }
            }

        } // extern "C"
    """

    "Return `size_rng` number of CUDA random number generator states."
    curr_gpu.make_context()
    gpu_vect_rand = ga.GPUArray((size_output,), dtype=np.int32)
    cpu_vect_rand = np.ones((size_output,), dtype=np.int32)
    (free, total) = cuda.mem_get_info()
    print(("Global memory occupancy:%f%% free" % (free * 100 / total)))

    # module = pycuda.compiler.SourceModule(init_rng_src, no_extern_c=True,arch="sm_30")
    module = pycuda.compiler.SourceModule(init_rng_src, no_extern_c=True)
    init_rng = module.get_function("init_rng")
    make_rand = module.get_function("make_rand")
    size_block = 1024
    n_blocks = size_output // size_block + 1
    rng_states = cuda.mem_alloc(
        n_blocks
        * characterize.sizeof("curandStateXORWOW", "#include <curand_kernel.h>")
    )
    init_rng(
        np.int32(n_blocks),
        rng_states,
        np.uint64(seed),
        np.uint64(0),
        block=(64, 1, 1),
        grid=(n_blocks // 64 + 1, 1),
    )
    try:
        make_rand(
            np.int32(size_output),
            rng_states,
            gpu_vect_rand,
            block=(size_block, 1, 1),
            grid=(n_blocks, 1),
        )
    except cuda.LogicError:
        print("random number generation failed ...")

    (free, total) = cuda.mem_get_info()
    print(("Global memory occupancy:%f%% free" % (free * 100 / total)))
    rng_states.free()
    gpu_vect_rand.get(ary=cpu_vect_rand)
    cuda.Context.pop()
    return cpu_vect_rand


def scn(input_mat, n_iter):
    """
    :param input_mat: matrice to normalize
    """
    scn_kernel = """

        __global__ void sum_along_axis(float* input, float* vect_sum, int width , int axis)
            {
                int r0 = threadIdx.x + blockDim.x * blockIdx.x;
                int r1 = threadIdx.y + blockDim.y * blockIdx.y;
                int coord = r0 * width + r1;
                if ((r0<width) && (r1< width)) {
                    float val = input[coord];
                    int id = (axis == 0) * r0 + (axis == 1) * r1;
                    val = atomicAdd( &(vect_sum[id]), (float) val);
                }
            }

        __global__ void norm_along_axis(float* input, float* vect_sum, int width, int axis)
            {
                int r0 = threadIdx.x + blockDim.x * blockIdx.x;
                int r1 = threadIdx.y + blockDim.y * blockIdx.y;
                int coord = r0 * width + r1;
                if ((r0<width) && (r1< width)) {
                    float val = input[coord];
                    int id = (axis == 0) * r0 + (axis == 1) * r1;
                    input[coord] = val / vect_sum[id];
                }
            }
        __global__ void init_vect_sum(float* vect_sum, int width)
            {
                int r0 = threadIdx.x + blockDim.x * blockIdx.x;
                if (r0 < width){
                    vect_sum[r0] = 0;
                }
            }
    """
    curr_gpu.make_context()
    #    if input_mat.dtype != np.float32:
    #        input_mat = np.float32(input_mat)
    (free, total) = cuda.mem_get_info()
    print(("Global memory occupancy:%f%% free" % (free * 100 / total)))
    if free > 2 * input_mat.nbytes:
        print(" gpu mem space ok ")

        width_mat = np.int32(input_mat.shape[0])
        try:
            n_elements = input_mat.shape[0] * input_mat.shape[1]
        except IndexError:
            n_elements = input_mat.shape[0]
        n_elements = np.int32(n_elements)
        cpu_vect_sum = np.zeros((width_mat,), dtype=np.float32)
        gpu_vect_sum = ga.to_gpu(cpu_vect_sum)
        gpu_data = ga.to_gpu(input_mat)
        cpu_output = np.empty_like(input_mat)

        # module = pycuda.compiler.SourceModule(scn_kernel,arch="sm_30")
        module = pycuda.compiler.SourceModule(scn_kernel)
        sum_along_axis = module.get_function("sum_along_axis")
        norm_along_axis = module.get_function("norm_along_axis")
        init_vect_sum = module.get_function("init_vect_sum")
        size_block_x = 32
        size_block_y = 32

        n_blocks_x = int(width_mat) // (size_block_x) + 1
        n_blocks_y = int(width_mat) // (size_block_y) + 1

        print("size block = ", size_block_x)
        print("n blocks x= ", n_blocks_x)
        print("n blocks y= ", n_blocks_y)

        print("gpu data type = ", gpu_data.dtype)
        print("gpu vect sum type = ", gpu_vect_sum.dtype)
        print("n element type = ", n_elements.dtype)
        print("width_mat type = ", width_mat.dtype)

        for i in range(0, n_iter):
            id_axis = np.int32(np.mod(i, 2))
            print("id_axis= ", id_axis)
            sum_along_axis(
                gpu_data,
                gpu_vect_sum,
                width_mat,
                id_axis,
                block=(size_block_x, size_block_y, 1),
                grid=(n_blocks_x, n_blocks_y),
                shared=0,
            )

            norm_along_axis(
                gpu_data,
                gpu_vect_sum,
                width_mat,
                id_axis,
                block=(size_block_x, size_block_y, 1),
                grid=(n_blocks_x, n_blocks_y),
                shared=0,
            )
            init_vect_sum(
                gpu_vect_sum, block=(64, 1, 1), grid=(int(width_mat) // 64 + 1, 1)
            )
        #            gpu_vect_sum = ga.to_gpu(cpu_vect_sum)

        gpu_data.get(ary=cpu_output)
        # gpu_data.free()
        cuda.Context.pop()
        return cpu_output


class randomize:
    def __init__(self, init_data, n_generators):

        self.ctx = curr_gpu.make_context()
        self.module = pycuda.compiler.SourceModule(kernels_cuda_src, no_extern_c=True)
        (free, total) = cuda.mem_get_info()
        print(("Global memory occupancy:%f%% free" % (free * 100 / total)))
        print(("Global free memory :%i Mo free" % (free / 10 ** 6)))

        ################################################################################################################

        self.width_mat = np.int32(init_data.shape[0])
        #        self.gpu_init_data = ga.to_gpu(init_data)
        self.gpu_init_data = cuda.mem_alloc(init_data.nbytes)
        cuda.memcpy_htod(self.gpu_init_data, init_data)

        self.cpu_new_data = np.zeros_like(init_data, dtype=np.float32)
        print("size new data = ", self.cpu_new_data.nbytes / 10 ** 6)
        (free, total) = cuda.mem_get_info()
        print(("Global memory occupancy:%f%% free" % (free * 100 / total)))
        print(("Global free memory :%i Mo free" % (free / 10 ** 6)))

        self.gpu_new_data = cuda.mem_alloc(self.cpu_new_data.nbytes)
        cuda.memcpy_htod(self.gpu_new_data, self.cpu_new_data)
        #        self.gpu_new_data = ga.to_gpu(self.cpu_new_data)

        self.cpu_vect_sum = np.zeros((self.width_mat,), dtype=np.float32)
        self.gpu_vect_sum = cuda.mem_alloc(self.cpu_vect_sum.nbytes)
        cuda.memcpy_htod(self.gpu_vect_sum, self.cpu_vect_sum)
        #        self.gpu_vect_sum = ga.to_gpu(self.cpu_vect_sum)
        ################################################################################################################
        self.init_rng = self.module.get_function("init_rng")
        self.gen_rand_mat = self.module.get_function("gen_rand_mat")
        self.sum_along_axis = self.module.get_function("sum_along_axis")
        self.norm_along_axis = self.module.get_function("norm_along_axis")
        self.init_vect_sum = self.module.get_function("init_vect_sum")
        self.copy_mat = self.module.get_function("copy_mat")
        ################################################################################################################
        self.n_generators = n_generators
        seed = 1
        self.rng_states = cuda.mem_alloc(
            n_generators
            * characterize.sizeof("curandStateXORWOW", "#include <curand_kernel.h>")
        )
        self.init_rng(
            np.int32(n_generators),
            self.rng_states,
            np.uint64(seed),
            np.uint64(0),
            block=(64, 1, 1),
            grid=(n_generators // 64 + 1, 1),
        )
        (free, total) = cuda.mem_get_info()

        size_block_x = 32
        size_block_y = 32
        n_blocks_x = int(self.width_mat) // (size_block_x) + 1
        n_blocks_y = int(self.width_mat) // (size_block_y) + 1
        self.grid = (n_blocks_x, n_blocks_y, 1)
        self.block = (size_block_x, size_block_y, 1)

    def generate_new_matrix(self, n_iter, do_random, fact_sub_sampling):
        #        fact_sub_sampling = np.float32(0.5)
        if do_random:
            self.gen_rand_mat(
                self.gpu_init_data,
                self.gpu_new_data,
                self.rng_states,
                np.int32(self.n_generators),
                self.width_mat,
                np.float32(fact_sub_sampling),
                block=self.block,
                grid=self.grid,
                shared=0,
            )
        else:
            self.copy_mat(
                self.gpu_init_data,
                self.gpu_new_data,
                self.width_mat,
                np.float32(fact_sub_sampling),
                block=self.block,
                grid=self.grid,
                shared=0,
            )
        print("matrix generated")
        for i in range(0, n_iter):
            #            print "iter,",i
            id_axis = np.int32(np.mod(i, 2))
            self.sum_along_axis(
                self.gpu_new_data,
                self.gpu_vect_sum,
                self.width_mat,
                id_axis,
                block=self.block,
                grid=self.grid,
                shared=0,
            )

            self.norm_along_axis(
                self.gpu_new_data,
                self.gpu_vect_sum,
                self.width_mat,
                id_axis,
                block=self.block,
                grid=self.grid,
                shared=0,
            )

            self.init_vect_sum(
                self.gpu_vect_sum,
                block=(64, 1, 1),
                grid=(int(self.width_mat) // 64 + 1, 1),
            )

        #        self.gpu_new_data.get(ary=self.cpu_new_data)
        cuda.memcpy_dtoh(self.cpu_new_data, self.gpu_new_data)
        return self.cpu_new_data

    def free_gpu(self,):
        self.rng_states.free()
        self.gpu_vect_sum.free()
        self.gpu_new_data.free()
        self.gpu_init_data.free()
        (free, total) = cuda.mem_get_info()
        print(
            (
                "Global memory occupancy after cleaning processes: %f%% free"
                % (free * 100 / total)
            )
        )
        print(("Global free memory :%i Mo free" % (free / 10 ** 6)))
        del self.module
        self.ctx.detach()


#        self.ctx.pop()
#        cuda.Context.pop()

if __name__ == "__main__":

    import numpy as np

    a = np.ones((10000, 10000), dtype=np.int32)
    gen = randomize(a, 1000)
    out = gen.generate_new_matrix(10)
    gen.free_gpu()
    print(out)
