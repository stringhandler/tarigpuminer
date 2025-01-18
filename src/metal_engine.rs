use std::os::raw::c_void;

use anyhow::Context;
use log::debug;
use metal::{
    ArgumentDescriptor,
    CompileOptions,
    ComputePassDescriptor,
    ComputePipelineDescriptor,
    Device,
    Function,
    Library,
    MTLResourceOptions,
    MTLSize,
};

use crate::{
    context_impl::ContextImpl,
    engine_impl::EngineImpl,
    function_impl::FunctionImpl,
    gpu_status_file::GpuStatus,
};

const NUM_SAMPLES: u64 = 2;
static LIBRARY_SRC: &str = include_str!("./metal_sha3.metal");

pub struct MetalContext {
    context: Device,
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
pub struct MetalEngine {}

impl EngineImpl for MetalEngine {
    type Context = MetalContext;
    type Function = MetalFunction;

    fn init(&mut self) -> Result<(), anyhow::Error> {
        debug!("MetalEngine: Initializing");
        Ok(())
    }

    fn num_devices(&self) -> Result<u32, anyhow::Error> {
        Ok(Device::all().len() as u32)
    }

    fn detect_devices(&self) -> Result<Vec<GpuStatus>, anyhow::Error> {
        let mut total_devices = 0;
        let mut gpu_devices: Vec<GpuStatus> = vec![];

        let all_devices = Device::all();

        for device in all_devices {
            let mut gpu_device = GpuStatus {
                device_name: device.name().to_string(),
                device_index: device.registry_id() as u32,
                is_available: true,
                max_grid_size: device.max_threadgroup_memory_length() as u32,
                grid_size: 0,
                block_size: 0,
            };

            let context = MetalContext { context: device };

            if let Ok(function) = self.create_main_function(&context).inspect_err(|error| {
                debug!("Failed to create main function: {:?}", error);
            }) {
                if let Ok((block_size, grid_size)) = function
                    .suggested_launch_configuration(&context.context)
                    .inspect_err(|error| {
                        debug!("Failed to get suggested launch configuration: {:?}", error);
                    })
                {
                    gpu_device.block_size = block_size;
                    gpu_device.grid_size = grid_size;
                }
                gpu_devices.push(gpu_device);
                total_devices += 1;
            }
        }

        if total_devices > 0 {
            return Ok(gpu_devices);
        }

        return Err(anyhow::anyhow!("No devices found"));
    }

    fn create_context(&self, device_index: u32) -> Result<Self::Context, anyhow::Error> {
        todo!()
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
        let kernel = function.program.get_function("sum", None).unwrap_or_else(|error| {
            panic!("Failed to get function sum: {:?}", error);
        });

        let command_queue = context.context.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();

        let counter_sample_buffer_desc = metal::CounterSampleBufferDescriptor::new();
        counter_sample_buffer_desc.set_storage_mode(metal::MTLStorageMode::Shared);
        counter_sample_buffer_desc.set_sample_count(NUM_SAMPLES);
        let counter_sets = context.context.counter_sets();

        let timestamp_counter = counter_sets.iter().find(|cs| cs.name() == "timestamp");

        counter_sample_buffer_desc.set_counter_set(timestamp_counter.expect("No timestamp counter found"));
        let counter_sample_buffer = context
            .context
            .new_counter_sample_buffer_with_descriptor(&counter_sample_buffer_desc)
            .unwrap();

        let compute_pass_descriptor = ComputePassDescriptor::new();

        let sample_buffer_attachment_descriptor = compute_pass_descriptor
            .sample_buffer_attachments()
            .object_at(0)
            .unwrap();

        sample_buffer_attachment_descriptor.set_sample_buffer(&counter_sample_buffer);
        sample_buffer_attachment_descriptor.set_start_of_encoder_sample_index(0);
        sample_buffer_attachment_descriptor.set_end_of_encoder_sample_index(1);

        let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

        let pipelione_state_descriptor = ComputePipelineDescriptor::new();
        pipelione_state_descriptor.set_compute_function(Some(&kernel));

        let pipeline_state = context
            .context
            .new_compute_pipeline_state_with_function(pipelione_state_descriptor.compute_function().unwrap())
            .unwrap();

        let sum_data = [
            1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
            29, 30,
        ];

        let data_buffer = context.context.new_buffer_with_data(
            unsafe { std::mem::transmute(sum_data.as_ptr()) },
            (sum_data.len() * std::mem::size_of::<u64>()) as u64,
            MTLResourceOptions::StorageModePrivate,
        );

        let output_buffer = {
            let data2 = [0u32];
            context.context.new_buffer_with_data(
                unsafe { std::mem::transmute(data2.as_ptr()) },
                (data2.len() * std::mem::size_of::<u32>()) as u64,
                MTLResourceOptions::CPUCacheModeDefaultCache,
            )
        };

        encoder.set_buffer(0, Some(&data_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);

        let num_threads = pipeline_state.thread_execution_width();

        let thread_group_count = MTLSize {
            width: grid_size as u64,
            height: 1,
            depth: 1,
        };

        let thread_group_size = MTLSize {
            width: num_threads,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let mut cpu_end: u64 = 0;
        let mut gpu_end: u64 = 0;

        context.context.sample_timestamps(&mut cpu_end, &mut gpu_end);

        let ptr = output_buffer.contents() as *mut u32;
        unsafe {
            println!("Output: {}", *ptr);
        };

        Ok((None, 0, 0))
    }
}

fn create_program_from_source(context: &Device) -> Result<Library, anyhow::Error> {
    debug!("MetalEngine: Creating program from source");
    let options = CompileOptions::new();

    let library = context
        .new_library_with_source(LIBRARY_SRC, &options)
        .unwrap_or_else(|error| {
            panic!("Failed to create library from source: {:?}", error);
        });

    Ok(library)
}
