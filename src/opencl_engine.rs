use std::{
    io::Read,
    ptr,
    sync::{Arc, RwLock},
};

use anyhow::Error;
use log::{error, info, warn};
use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE},
    context::Context,
    device::{Device, CL_DEVICE_TYPE_GPU},
    kernel::{ExecuteKernel, Kernel},
    memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY},
    platform::{get_platforms, Platform},
    program::Program,
    types::{cl_ulong, CL_TRUE},
};

use crate::{
    context_impl::ContextImpl, engine_impl::EngineImpl, function_impl::FunctionImpl, gpu_status_file::GpuStatus,
};

const LOG_TARGET: &str = "tari::gpuminer::opencl";

pub struct OpenClEngineInner {
    platforms: Vec<Platform>,
}

#[derive(Clone)]
pub struct OpenClEngine {
    inner: Arc<RwLock<OpenClEngineInner>>,
}

impl OpenClEngine {
    pub fn new() -> Self {
        OpenClEngine {
            inner: Arc::new(RwLock::new(OpenClEngineInner { platforms: vec![] })),
        }
    }
}

impl EngineImpl for OpenClEngine {
    type Context = OpenClContext;
    type Function = OpenClFunction;

    fn init(&mut self) -> Result<(), anyhow::Error> {
        info!(target: LOG_TARGET, "OpenClEngine: init engine");
        let platforms = get_platforms()?;
        let mut lock = self.inner.write().unwrap();
        lock.platforms = platforms;
        Ok(())
    }

    fn num_devices(&self) -> Result<u32, anyhow::Error> {
        let mut total_devices = 0;
        let lock = self.inner.read().unwrap();
        for platform in lock.platforms.iter() {
            let devices = platform.get_devices(CL_DEVICE_TYPE_GPU)?;
            total_devices += devices.len();
        }
        info!(target: LOG_TARGET, "OpenClEngine: total number of devices {:?}", total_devices);
        Ok(total_devices as u32)
    }

    fn detect_devices(&self) -> Result<Vec<GpuStatus>, anyhow::Error> {
        let mut total_devices = 0;
        let mut gpu_devices: Vec<GpuStatus> = vec![];
        let lock = self.inner.read().unwrap();
        for platform in lock.platforms.iter() {
            let devices = platform.get_devices(CL_DEVICE_TYPE_GPU)?;

            info!(target: LOG_TARGET, "OpenClEngine: platform name: {}", platform.name()?);
            println!("List of the devices for the Platform: {}", platform.name()?);
            for device in devices {
                let dev = Device::new(device);
                let name = dev.name().unwrap_or_default() as String;
                info!(target: LOG_TARGET, "Device index {:?}: {}", total_devices, &name);
                println!("device: {}", &name);
                let mut gpu = GpuStatus {
                    device_name: name,
                    is_available: true,
                };
                gpu_devices.push(gpu);
                total_devices += 1;
            }
        }
        if total_devices > 0 {
            return Ok(gpu_devices);
        }
        return Err(anyhow::anyhow!("No gpu device detected"));
    }

    fn create_context(&self, device_index: u32) -> Result<Self::Context, anyhow::Error> {
        info!(target: LOG_TARGET, "OpenClEngine: create context");
        let lock = self.inner.write().unwrap();
        let mut devices = vec![];
        for platform in lock.platforms.iter() {
            devices.extend_from_slice(&platform.get_devices(CL_DEVICE_TYPE_GPU)?);
        }
        let device = devices[device_index as usize];
        let context = Context::from_device(&Device::new(device))?;
        Ok(OpenClContext::new(context))
    }

    fn create_main_function(&self, context: &Self::Context) -> Result<Self::Function, anyhow::Error> {
        info!(target: LOG_TARGET, "OpenClEngine: create function");
        let program = create_program_from_source(&context.context).unwrap();
        Ok(OpenClFunction { program })
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
    ) -> Result<(Option<u64>, u32, u64), Error> {
        // TODO: put in multiple threads

        let kernels = vec![Kernel::create(&function.program, "sha3").expect("bad kernel")];

        //  let queue = CommandQueue::create_default_with_properties(
        //     &context.context,
        //     CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
        //     0
        // )?;
        unsafe {
            let queue = CommandQueue::create_default(&context.context, CL_QUEUE_PROFILING_ENABLE)
                .expect("could not create command queue");

            let batch_size = 1 << 19; // According to tests, but we can try work this out
            let global_dimensions = [batch_size as usize];
            let max_workgroups = Device::new(context.context.devices()[0]).max_work_group_size().unwrap();
            // dbg!(max_compute);
            // let max_work_items = queue.max_work_item_dimensions();
            // dbg!(max_work_items);
            // dbg!("here");
            let mut buffer =
                Buffer::<cl_ulong>::create(&context.context, CL_MEM_READ_ONLY, data.len(), ptr::null_mut())?;
            queue.enqueue_write_buffer(&mut buffer, CL_TRUE, 0, data, &[])?;
            let output_buffer = Buffer::<cl_ulong>::create(&context.context, CL_MEM_WRITE_ONLY, 2, ptr::null_mut())?;
            // dbg!(block_size);
            // dbg!(grid_size);
            for kernel in kernels {
                ExecuteKernel::new(&kernel)
            .set_arg(&buffer)
            .set_arg(&nonce_start)
            .set_arg(&min_difficulty)
            .set_arg(&num_iterations)
            .set_arg(&output_buffer)

            .set_global_work_size((grid_size * block_size) as usize)
            // .set_local_work_size(max_workgroups)
            // .set_wait_event(&y_write_event)
            .enqueue_nd_range(&queue).expect("culd not queue");

                // TODO: find out better workdim
                // queue.enqueue_nd_range_kernel(kernel.get(), 1, 0 as *const usize, global_dimensions.as_ptr(), 0 as
                // *const usize, &[]).expect("could not execute");
            }
            queue.finish()?;
            let mut output = vec![0u64, 0u64];
            queue.enqueue_read_buffer(&output_buffer, CL_TRUE, 0, output.as_mut_slice(), &[])?;
            if output[0] > 0 {
                return Ok((
                    Some(output[0]),
                    grid_size * block_size * num_iterations,
                    u64::MAX / output[1],
                ));
            }
            return Ok((None, grid_size * block_size * num_iterations, u64::MAX / output[1]));
        }
        Ok((None, grid_size * block_size * num_iterations, 0))
    }
}
fn create_program_from_source(context: &Context) -> Option<Program> {
    let opencl_code = include_str!("./opencl_sha3.cl");
    info!(target: LOG_TARGET, "OpenClEngine: create program from source");
    // Load the program from file.
    let mut program = match Program::create_from_source(&context, &opencl_code) {
        Ok(program) => {
            info!(target: LOG_TARGET, "OpenClEngine: program created successfully");
            program
        },
        Err(error) => {
            error!(target: LOG_TARGET, "OpenClEngine: program creating error : {}", error);
            println!("Programing creating error : {}", error);
            unimplemented!("");
        },
    };

    // Build the program.
    match program.build(context.devices(), "") {
        Ok(_) => {
            info!(target: LOG_TARGET, "OpenClEngine: program built successfully");
            Some(program)
        },
        Err(error) => {
            error!(target: LOG_TARGET, "OpenClEngine: program building error : {}", error);
            println!("Program building error : {}", error);
            for device_id in context.devices() {
                match program.get_build_log(*device_id) {
                    Ok(log) => {
                        info!(target: LOG_TARGET, "OpenClEngine: program log {}", log);
                        println!("{}", log)
                    },
                    Err(error) => {
                        error!(target: LOG_TARGET, "OpenClEngine: error getting the build log : {}", error);
                        println!("Error getting the build log : {}", error)
                    },
                };
            }
            None
        },
    }
}

pub struct OpenClContext {
    context: Context,
}

impl OpenClContext {
    pub fn new(context: Context) -> Self {
        OpenClContext { context }
    }
}

impl ContextImpl for OpenClContext {}

pub struct OpenClFunction {
    program: Program,
}
impl FunctionImpl for OpenClFunction {
    fn suggested_launch_configuration(&self) -> Result<(u32, u32), anyhow::Error> {
        Ok((1000, 1000))
    }
}
