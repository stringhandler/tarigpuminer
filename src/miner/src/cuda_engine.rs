
#[cfg(feature = "nvidia")]
use cust::{
    device::DeviceAttribute,
    memory::{AsyncCopyDestination, DeviceCopy},
    module::{ModuleJitOption, ModuleJitOption::DetermineTargetFromContext},
    prelude::{Module, *},
};
pub struct CudaEngine {

}

impl CudaEngine {

    pub fn init() -> Result<(), anyhow::Error> {
        #[cfg(feature = "nvidia")]
        cust::init(CudaFlags::empty())?;
    }


    fn mine<T: DeviceCopy>(
        mining_hash: FixedHash,
        pow: Vec<u8>,
        target: u64,
        nonce_start: u64,
        context: &Context,
        module: &Module,
        num_iterations: u32,
        func: &Function<'_>,
        block_size: u32,
        grid_size: u32,
        data_ptr: DevicePointer<T>,
        output_buf: &DeviceBuffer<u64>,
    ) -> Result<(Option<u64>, u32, u64), anyhow::Error> {
        let num_streams = 1;
        let mut streams = Vec::with_capacity(num_streams);
        let mut max = None;

        let output = vec![0u64; 5];

        let timer = Instant::now();
        for st in 0..num_streams {
            let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

            streams.push(stream);
        }

        for st in 0..num_streams {
            let stream = &streams[st];
            unsafe {
                launch!(
                func<<<grid_size, block_size, 0, stream>>>(
                data_ptr,
                     nonce_start,
                     target,
                     num_iterations,
                     output_buf.as_device_ptr(),

                )
            )?;
            }
        }

        for st in 0..num_streams {
            let mut out1 = vec![0u64; 5];

            unsafe {
                output_buf.copy_to(&mut out1)?;
            }
            // stream.synchronize()?;

            if out1[0] > 0 {
                return Ok((Some((&out1[0]).clone()), grid_size * block_size * num_iterations, 0));
            }
        }

        match max {
            Some((i, diff)) => {
                if diff > target {
                    return Ok((
                        Some(i),
                        grid_size * block_size * num_iterations * num_streams as u32,
                        diff,
                    ));
                }
                return Ok((None, grid_size * block_size * num_iterations * num_streams as u32, diff));
            },
            None => Ok((None, grid_size * block_size * num_iterations * num_streams as u32, 0)),
        }
    }
}

