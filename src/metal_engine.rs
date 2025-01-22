use std::{mem::transmute, os::raw::c_void};

use anyhow::Context;
use log::{debug, info};
use metal::{
    objc::rc::autoreleasepool, ArgumentDescriptor, Array, CompileOptions, ComputePassDescriptor, ComputePipelineDescriptor, Device, Function, Library, MTLResourceOptions, MTLResourceUsage, MTLSize, NSUInteger, Texture
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
        let kernel = self.program.get_function("sum", None).unwrap_or_else(|error| {
            panic!("Failed to get function sum: {:?}", error);
        });

        let block_size = kernel.device().recommended_max_working_set_size() as u32;
        let grid_size = kernel.device().max_threadgroup_memory_length() as u32;

        Ok((block_size, grid_size))
    }
}

#[derive(Clone)]
pub struct MetalEngine {
    sum: u32,
}
impl MetalEngine {
    pub fn new() -> Self {
        MetalEngine {
            sum: 0,
        }
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
                debug!(target: LOG_TARGET,"Failed to create context: {:?}", error);
            }) {
                if let Ok(function) = self.create_main_function(&context).inspect_err(|error| {
                    debug!(target: LOG_TARGET,"Failed to create main function: {:?}", error);
                }) {
                    if let Ok((block_size, grid_size)) = function
                        .suggested_launch_configuration(&context.context)
                        .inspect_err(|error| {
                            debug!(target: LOG_TARGET,"Failed to get suggested launch configuration: {:?}", error);
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
        let device = all_devices.get(device_index as usize).unwrap_or_else(|| {
            panic!("Failed to get device with index: {}", device_index);
        });

        Ok(MetalContext::new(device.clone()))
    }

    fn create_main_function(&self, context: &Self::Context) -> Result<Self::Function, anyhow::Error> {
        let function = create_program_from_source(&context.context).unwrap_or_else(|error| {
            panic!("Failed to create program from source: {:?}", error);
        });
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
        info!(target: LOG_TARGET,"MetalEngine: Mining");
        info!(target: LOG_TARGET,"Data: {:?}", data);
        info!(target: LOG_TARGET,"Min difficulty: {}", min_difficulty);
        info!(target: LOG_TARGET,"Nonce start: {}", nonce_start);
        info!(target: LOG_TARGET,"Num iterations: {}", num_iterations);

      autoreleasepool(|| {
        let command_queue = context.context.new_command_queue();

        let buffer = context.context.new_buffer_with_data(
            unsafe { transmute(data.as_ptr()) },
            (data.len() * size_of::<u32>()) as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );

        info!(target: LOG_TARGET,"Creating constants input buffer");

        let min_difficulty = {
            let min_difficulty_data = [min_difficulty];
            context.context.new_buffer_with_data(
                unsafe { transmute(min_difficulty_data.as_ptr()) },
                (min_difficulty_data.len() * size_of::<u32>()) as u64,
                MTLResourceOptions::CPUCacheModeDefaultCache,
            )
        };

        let nonce_start = {
            let nonce_start_data = [nonce_start];
            context.context.new_buffer_with_data(
                unsafe { transmute(nonce_start_data.as_ptr()) },
                (nonce_start_data.len() * size_of::<u32>()) as u64,
                MTLResourceOptions::CPUCacheModeDefaultCache,
            )
        };

        let num_iterations = {
            let num_iterations_data = [num_iterations];
            context.context.new_buffer_with_data(
                unsafe { transmute(num_iterations_data.as_ptr()) },
                (num_iterations_data.len() * size_of::<u32>()) as u64,
                MTLResourceOptions::CPUCacheModeDefaultCache,
            )
        };

        info!(target: LOG_TARGET,"Creating sum buffer");

        let output = {
            let output_data =  vec![0u64, 0u64];
            context.context.new_buffer_with_data(
                unsafe { transmute(output_data.as_ptr()) },
                (output_data.len() * size_of::<u32>()) as u64,
                MTLResourceOptions::CPUCacheModeDefaultCache,
            )
        };


        let min_difficulty_argument_descriptor = ArgumentDescriptor::new();
        min_difficulty_argument_descriptor.set_index(0);
        min_difficulty_argument_descriptor.set_data_type(metal::MTLDataType::ULong);

        let nonce_start_argument_descriptor = ArgumentDescriptor::new();
        nonce_start_argument_descriptor.set_index(1);
        nonce_start_argument_descriptor.set_data_type(metal::MTLDataType::ULong);

        let num_iterations_argument_descriptor = ArgumentDescriptor::new();
        num_iterations_argument_descriptor.set_index(2);
        num_iterations_argument_descriptor.set_data_type(metal::MTLDataType::UInt);

        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        let kernel = function.program.get_function("sha3",None).unwrap();

        // info!(target: LOG_TARGET,"Creating argument encoder: {:?}", context.context.argument_buffers_support());

        let argument_encoder = context.context.new_argument_encoder(Array::from_slice(
            &[
                min_difficulty_argument_descriptor,
                nonce_start_argument_descriptor,
                num_iterations_argument_descriptor,
            ],
        ));
        // let argument_encoder = kernel.new_argument_encoder(0);

        info!(target: LOG_TARGET,"Creating argument buffer: {:?}", argument_encoder.encoded_length());

        let arg_buffer = context.context.new_buffer(
            argument_encoder.encoded_length(),
            MTLResourceOptions::empty(),
        );

        info!(target: LOG_TARGET,"Setting argument buffer: {:?}", arg_buffer.length());

        argument_encoder.set_argument_buffer(&arg_buffer, 0);
        info!(target: LOG_TARGET,"Setting buffer nonce_start");
        argument_encoder.set_buffer(0, &nonce_start, 0);
        info!(target: LOG_TARGET,"Setting buffer min_difficulty");
        argument_encoder.set_buffer(1, &min_difficulty,1);
        info!(target: LOG_TARGET,"Setting buffer num_iterations");
        argument_encoder.set_buffer(2, &num_iterations, 2);

        info!(target: LOG_TARGET,"Setting pipeline state descriptor");
        let pipeline_state_descriptor = ComputePipelineDescriptor::new();
        pipeline_state_descriptor.set_compute_function(Some(&kernel));

        info!(target: LOG_TARGET,"Creating pipeline state");

        let pipeline_state = context.context
            .new_compute_pipeline_state_with_function(
                pipeline_state_descriptor.compute_function().unwrap(),
            )
            .unwrap();

        info!(target: LOG_TARGET,"Setting compute pipeline state");

        encoder.set_compute_pipeline_state(&pipeline_state);
        encoder.set_buffer(0, Some(&arg_buffer), 0);
        encoder.set_buffer(1, Some(&buffer), 0);
        encoder.set_buffer(2, Some(&output), 0);

        info!(target: LOG_TARGET,"Using resources");

        encoder.use_resource(&buffer, MTLResourceUsage::Read);
        encoder.use_resource(&arg_buffer, MTLResourceUsage::Read);
        encoder.use_resource(&output, MTLResourceUsage::Write);

        let threads_per_thread_group = MTLSize {
            width: 16,
            height: 1,
            depth: 1,
        };

        let threads_per_grid = MTLSize {
            width: (data.len() as f64 / threads_per_thread_group.width as f64).ceil() as u64,
            height: 1,
            depth: 1,
        };

        info!(target: LOG_TARGET,"Threads per thread group: {:?}", threads_per_thread_group);
        info!(target: LOG_TARGET,"Threads per grid: {:?}", threads_per_grid);

        encoder.dispatch_thread_groups(threads_per_grid, threads_per_thread_group);
            

        // let width = 16;

        // let thread_group_count = MTLSize {
        //     width,
        //     height: 1,
        //     depth: 1,
        // };

        // let thread_group_size = MTLSize {
        //     width: (data.len() as u64 + width) / width,
        //     height: 1,
        //     depth: 1,
        // };

        // info!(target: LOG_TARGET,"Thread group size: {:?}", thread_group_size);
        // info!(target: LOG_TARGET,"Thread group count: {:?}", thread_group_count);

        // encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        
        encoder.end_encoding();
        info!(target: LOG_TARGET,"Command buffer commit");
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let ptr = output.contents() as *mut u32;
        unsafe {
            println!("Sum: {}", *ptr);
            Ok((None, *ptr, 0))
        }
    })

    }
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
