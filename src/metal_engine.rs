use std::mem::{size_of, transmute};

use log::{debug, error};
use metal::{
    objc::rc::autoreleasepool,
    Buffer,
    CompileOptions,
    ComputePipelineDescriptor,
    ComputePipelineState,
    Device,
    Function,
    Library,
    MTLResourceOptions,
    MTLResourceUsage,
    MTLSize,
};

use crate::{
    context_impl::ContextImpl,
    engine_impl::EngineImpl,
    function_impl::FunctionImpl,
    gpu_status_file::GpuStatus,
};

const LOG_TARGET: &str = "tari::gpuminer::metal";
static LIBRARY_SRC: &str = include_str!("./metal_sha3.metal");

pub struct MetalContext {
    context: Device,
}
impl MetalContext {
    pub fn new(context: Device) -> Self {
        MetalContext { context }
    }
}
impl ContextImpl for MetalContext {}

pub struct MetalFunction {
    program: Library,
}
impl FunctionImpl for MetalFunction {
    type Device = Device;

    fn suggested_launch_configuration(&self, device: &Self::Device) -> Result<(u32, u32), anyhow::Error> {
        let kernel = self.program.get_function("sha3", None).unwrap_or_else(|error| {
            panic!("Failed to get function sum: {:?}", error);
        });

        let block_size = kernel.device().recommended_max_working_set_size() as u32;
        let grid_size = kernel.device().max_threads_per_threadgroup().width as u32;

        Ok((block_size, grid_size))
    }
}

#[derive(Clone)]
pub struct MetalEngine {}
impl MetalEngine {
    pub fn new() -> Self {
        MetalEngine {}
    }
}

impl EngineImpl for MetalEngine {
    type Context = MetalContext;
    type Function = MetalFunction;

    fn init(&mut self) -> Result<(), anyhow::Error> {
        debug!(target: LOG_TARGET,"MetalEngine: Initializing");
        Ok(())
    }

    fn num_devices(&self) -> Result<u32, anyhow::Error> {
        Ok(Device::all().len() as u32)
    }

    fn detect_devices(&self) -> Result<Vec<GpuStatus>, anyhow::Error> {
        let mut total_devices = 0;
        let mut gpu_devices: Vec<GpuStatus> = vec![];

        let all_devices = Device::all();

        for (id, device) in all_devices.into_iter().enumerate() {
            let mut gpu_device = GpuStatus {
                device_name: device.name().to_string(),
                device_index: id as u32,
                is_available: true,
                max_grid_size: device.max_threadgroup_memory_length() as u32,
                grid_size: 0,
                block_size: 0,
            };

            if let Ok(context) = self.create_context(gpu_device.device_index).inspect_err(|error| {
                error!(target: LOG_TARGET,"Failed to create context: {:?}", error);
            }) {
                if let Ok(function) = self.create_main_function(&context).inspect_err(|error| {
                    error!(target: LOG_TARGET,"Failed to create main function: {:?}", error);
                }) {
                    if let Ok((block_size, grid_size)) = function
                        .suggested_launch_configuration(&context.context)
                        .inspect_err(|error| {
                            error!(target: LOG_TARGET,"Failed to get suggested launch configuration: {:?}", error);
                        })
                    {
                        gpu_device.block_size = block_size;
                        gpu_device.grid_size = grid_size;
                    }
                    gpu_devices.push(gpu_device);
                    total_devices += 1;
                }
            }
        }

        if total_devices > 0 {
            return Ok(gpu_devices);
        }

        return Err(anyhow::anyhow!("No devices found"));
    }

    fn create_context(&self, device_index: u32) -> Result<Self::Context, anyhow::Error> {
        let all_devices = Device::all();
        let device = all_devices
            .get(device_index as usize)
            .ok_or(anyhow::anyhow!("create_context: Device not found"))?;

        Ok(MetalContext::new(device.clone()))
    }

    fn create_main_function(&self, context: &Self::Context) -> Result<Self::Function, anyhow::Error> {
        let function = create_program_from_source(&context.context)?;
        Ok(MetalFunction { program: function })
    }

    fn mine(
        &self,
        function: &Self::Function,
        context: &Self::Context,
        data: &[u64],
        min_difficulty: u64,
        nonce_start: u64,
        num_iterations: u32,
        block_size: u32,
        grid_size: u32,
    ) -> Result<(Option<u64>, u32, u64), anyhow::Error> {
        autoreleasepool(|| {
            let command_queue = context.context.new_command_queue();

            let (buffer, min_difficulty_buffer, nonce_start_buffer, num_iterations_buffer, output) =
                create_buffers(&context.context, data, min_difficulty, nonce_start, num_iterations);

            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            let kernel = function
                .program
                .get_function("sha3", None)
                .map_err(|error| anyhow::anyhow!("Failed to get function sha3: {:?}", error))?;

            let pipeline_state = create_pipeline_state(&context.context, &kernel)?;
            encoder.set_compute_pipeline_state(&pipeline_state);

            debug!(target: LOG_TARGET,"Setting buffers for arguments");
            encoder.set_buffer(0, Some(&buffer), 0);
            encoder.set_buffer(1, Some(&output), 0);
            encoder.set_buffer(2, Some(&nonce_start_buffer), 0);
            encoder.set_buffer(3, Some(&min_difficulty_buffer), 0);
            encoder.set_buffer(4, Some(&num_iterations_buffer), 0);

            debug!(target: LOG_TARGET,"Describing resources");
            encoder.use_resource(&buffer, MTLResourceUsage::Read);
            encoder.use_resource(&min_difficulty_buffer, MTLResourceUsage::Read);
            encoder.use_resource(&nonce_start_buffer, MTLResourceUsage::Read);
            encoder.use_resource(&num_iterations_buffer, MTLResourceUsage::Read);
            encoder.use_resource(&output, MTLResourceUsage::Write);
            encoder.memory_barrier_with_resources(&[&output]);

            let threads = (block_size / 2).min(grid_size / 2) as u64;

            let threads_per_thread_group = MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            };

            let threads_per_grid = MTLSize {
                width: (data.len() as f64 / threads_per_thread_group.width as f64).ceil() as u64,
                height: 1,
                depth: 1,
            };

            debug!(target: LOG_TARGET,"Threads per thread group: {:?}", threads_per_thread_group);
            debug!(target: LOG_TARGET,"Threads per grid: {:?}", threads_per_grid);

            encoder.dispatch_threads(threads_per_grid, threads_per_thread_group);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            let ptr = output.contents() as *mut [u64; 2];
            unsafe {
                let result = *ptr;
                if result[0] > 0 {
                    return Ok((Some(result[0]), block_size * grid_size * num_iterations, u64::MAX / result[1]));
                } else {
                    return Ok((None, block_size * grid_size * num_iterations, u64::MAX / result[1]));
                }
            }
        })
    }
}

fn create_buffers(
    context: &Device,
    data: &[u64],
    min_difficulty: u64,
    nonce_start: u64,
    num_iterations: u32,
) -> (Buffer, Buffer, Buffer, Buffer, Buffer) {
    debug!(target: LOG_TARGET,"MetalEngine: Creating buffers");

    debug!(target: LOG_TARGET,"Creating data buffer with data: {:?}", data);
    let data_buffer = context.new_buffer_with_data(
        unsafe { transmute(data.as_ptr()) },
        (data.len() * size_of::<u32>()) as u64,
        MTLResourceOptions::CPUCacheModeDefaultCache,
    );

    debug!(target: LOG_TARGET,"Creating min_difficulty buffer with data: {:?}", min_difficulty);
    let min_difficulty_buffer = {
        let min_difficulty_data = [min_difficulty];
        context.new_buffer_with_data(
            unsafe { transmute(min_difficulty_data.as_ptr()) },
            (min_difficulty_data.len() * size_of::<u64>()) as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        )
    };

    debug!(target: LOG_TARGET,"Creating nonce_start buffer with data: {:?}", nonce_start);
    let nonce_start_buffer = {
        let nonce_start_data = [nonce_start];
        context.new_buffer_with_data(
            unsafe { transmute(nonce_start_data.as_ptr()) },
            (nonce_start_data.len() * size_of::<u64>()) as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        )
    };

    debug!(target: LOG_TARGET,"Creating num_iterations buffer with data: {:?}", num_iterations);
    let num_iterations_buffer = {
        let num_iterations_data = [num_iterations];
        context.new_buffer_with_data(
            unsafe { transmute(num_iterations_data.as_ptr()) },
            (num_iterations_data.len() * size_of::<u32>()) as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        )
    };

    debug!(target: LOG_TARGET,"Creating output buffer");
    let output = {
        let output_data = vec![0u64, 0u64];
        context.new_buffer_with_data(
            unsafe { transmute(output_data.as_ptr()) },
            (output_data.len() * size_of::<u64>()) as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        )
    };

    (
        data_buffer,
        min_difficulty_buffer,
        nonce_start_buffer,
        num_iterations_buffer,
        output,
    )
}

fn create_pipeline_state(device: &Device, kernel: &Function) -> Result<ComputePipelineState, anyhow::Error> {
    debug!(target: LOG_TARGET,"MetalEngine: Creating pipeline state");
    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor.set_compute_function(Some(&kernel));

    let compute_function = pipeline_state_descriptor
        .compute_function()
        .ok_or_else(|| anyhow::anyhow!("Failed to get compute function from pipeline state descriptor"))?;

    let pipeline_state = device
        .new_compute_pipeline_state_with_function(compute_function)
        .map_err(|error| anyhow::anyhow!("Failed to create compute pipeline state: {:?}", error))?;

    Ok(pipeline_state)
}

fn create_program_from_source(context: &Device) -> Result<Library, anyhow::Error> {
    debug!(target: LOG_TARGET,"MetalEngine: Creating program from source");
    let options = CompileOptions::new();

    let library = context
        .new_library_with_source(LIBRARY_SRC, &options)
        .unwrap_or_else(|error| {
            panic!("Failed to create library from source: {:?}", error);
        });

    Ok(library)
}
