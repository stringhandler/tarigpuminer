use std::{cmp, convert::TryInto, env::current_dir, fs, path::PathBuf, str::FromStr, sync::Arc, thread, time::Instant};

use anyhow::{anyhow, Context as AnyContext};
use clap::Parser;
#[cfg(feature = "nvidia")]
use cust::{
    memory::{AsyncCopyDestination, DeviceCopy},
    prelude::*,
};
use log::{error, info, warn};
use minotari_app_grpc::tari_rpc::{
    Block, BlockHeader as grpc_header, NewBlockTemplate, TransactionOutput as GrpcTransactionOutput,
};
use num_format::{Locale, ToFormattedString};
use sha3::Digest;
use tari_common::{configuration::Network, initialize_logging};
use tari_common_types::{tari_address::TariAddress, types::FixedHash};
use tari_core::{
    blocks::BlockHeader,
    consensus::ConsensusManager,
    transactions::{
        key_manager::create_memory_db_key_manager, tari_amount::MicroMinotari, transaction_components::RangeProofType,
    },
};
use tari_shutdown::Shutdown;
use tokio::{runtime::Runtime, sync::RwLock};

#[cfg(feature = "nvidia")]
use crate::cuda_engine::CudaEngine;
#[cfg(feature = "opencl3")]
use crate::opencl_engine::OpenClEngine;
use crate::{
    config_file::ConfigFile,
    engine_impl::EngineImpl,
    function_impl::FunctionImpl,
    gpu_engine::GpuEngine,
    gpu_status_file::GpuStatusFile,
    http::{config::Config, server::HttpServer},
    node_client::{ClientType, NodeClient},
    stats_store::StatsStore,
    tari_coinbase::generate_coinbase,
};

use tari_core::transactions::transaction_components::CoinBaseExtra;

mod config_file;
mod context_impl;
#[cfg(feature = "nvidia")]
mod cuda_engine;
mod engine_impl;
mod function_impl;
mod gpu_engine;
mod gpu_status_file;
mod http;
mod node_client;
#[cfg(feature = "opencl3")]
mod opencl_engine;
mod p2pool_client;
mod stats_store;
mod tari_coinbase;

const LOG_TARGET: &str = "tari::gpuminer";

#[tokio::main]
async fn main() {
    match main_inner().await {
        Ok(()) => {
            info!(target: LOG_TARGET, "Gpu miner startup process completed successfully");
            std::process::exit(0);
        },
        Err(err) => {
            error!(target: LOG_TARGET, "Gpu miner startup process error: {}", err);
            std::process::exit(1);
        },
    }
}

#[derive(Parser)]
struct Cli {
    /// Config file path
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Do benchmark
    #[arg(short, long)]
    benchmark: bool,

    /// (Optional) Tari wallet address to send rewards to
    #[arg(short = 'a', long)]
    tari_address: Option<String>,

    /// (Optional) Tari base node/p2pool node URL
    #[arg(short = 'u', long)]
    tari_node_url: Option<String>,

    /// P2Pool enabled
    #[arg(long)]
    p2pool_enabled: bool,

    /// Enable/disable http server
    ///
    /// It exposes health-check, version and stats endpoints
    #[arg(long)]
    http_server_enabled: Option<bool>,

    /// Port of HTTP server
    #[arg(long)]
    http_server_port: Option<u16>,

    /// GPU percentage in values 1-1000, where 500 = 50%
    #[arg(long, alias = "gpu-usage")]
    gpu_percentage: Option<u16>,

    /// grid_size for the gpu
    #[arg(long, alias = "grid-size")]
    grid_size: Option<u32>,
    /// Coinbase extra data
    #[arg(long)]
    coinbase_extra: Option<String>,

    /// (Optional) log config file
    #[arg(long, alias = "log-config-file", value_name = "log-config-file")]
    log_config_file: Option<PathBuf>,

    /// (Optional) log dir
    #[arg(long, alias = "log-dir", value_name = "log-dir")]
    log_dir: Option<PathBuf>,

    /// (Optional) log dir
    #[arg(short = 'd', long, alias = "detect")]
    detect: Option<bool>,

    /// (Optional) use only specific devices
    #[arg(long, alias = "use-devices", num_args=0.., value_delimiter=',')]
    use_devices: Option<Vec<u32>>,

    /// (Optional) exclude specific devices from use
    #[arg(long, alias = "exclude-devices", num_args=0.., value_delimiter=',')]
    exclude_devices: Option<Vec<u32>>,

    /// Gpu status file path
    #[arg(short, long, value_name = "gpu-status")]
    gpu_status_file: Option<PathBuf>,
}

async fn main_inner() -> Result<(), anyhow::Error> {
    let cli = Cli::parse();
    info!(target: LOG_TARGET, "Xtrgpuminer init");
    if let Some(ref log_dir) = cli.log_dir {
        tari_common::initialize_logging(
            &log_dir.join("log4rs_config.yml"),
            &log_dir.join("xtrgpuminer"),
            include_str!("../log4rs_sample.yml"),
        )
        .expect("Could not set up logging");
    }

    let benchmark = cli.benchmark;
    let mut config = match ConfigFile::load(&cli.config.as_ref().cloned().unwrap_or_else(|| {
        let mut path = current_dir().expect("no current directory");
        path.push("config.json");
        path
    })) {
        Ok(config) => {
            info!(target: LOG_TARGET, "Config file loaded successfully");
            config
        },
        Err(err) => {
            eprintln!("Error loading config file: {}. Creating new one", err);
            let default = ConfigFile::default();
            let path = cli.config.unwrap_or_else(|| {
                let mut path = current_dir().expect("no current directory");
                path.push("config.json");
                path
            });
            dbg!(&path);
            fs::create_dir_all(path.parent().expect("no parent"))?;
            default.save(&path).expect("Could not save default config");
            default
        },
    };

    if let Some(ref addr) = cli.tari_address {
        config.tari_address = addr.clone();
    }
    if let Some(ref url) = cli.tari_node_url {
        config.tari_node_url = url.clone();
    }
    if cli.p2pool_enabled {
        config.p2pool_enabled = true;
    }
    if let Some(enabled) = cli.http_server_enabled {
        config.http_server_enabled = enabled;
    }
    if let Some(port) = cli.http_server_port {
        config.http_server_port = port;
    }
    if let Some(percentage) = cli.gpu_percentage {
        config.gpu_percentage = percentage;
    }
    if let Some(grid_size) = cli.grid_size {
        config.grid_size = grid_size;
    }
    if let Some(coinbase_extra) = cli.coinbase_extra {
        config.coinbase_extra = coinbase_extra;
    }

    let submit = true;

    #[cfg(feature = "nvidia")]
    let mut gpu_engine = GpuEngine::new(CudaEngine::new());

    #[cfg(feature = "opencl3")]
    let mut gpu_engine = GpuEngine::new(OpenClEngine::new());

    gpu_engine.init().unwrap();

    // http server
    let mut shutdown = Shutdown::new();
    let stats_store = Arc::new(StatsStore::new());
    if config.http_server_enabled {
        let http_server_config = Config::new(config.http_server_port);
        info!(target: LOG_TARGET, "HTTP server runs on port: {}", &http_server_config.port);
        let http_server = HttpServer::new(shutdown.to_signal(), http_server_config, stats_store.clone());
        info!(target: LOG_TARGET, "HTTP server enabled");
        tokio::spawn(async move {
            if let Err(error) = http_server.start().await {
                println!("Failed to start HTTP server: {error:?}");
                error!(target: LOG_TARGET, "Failed to start HTTP server: {:?}", error);
            } else {
                info!(target: LOG_TARGET, "Success to start HTTP server");
            }
        });
    }

    let num_devices = gpu_engine.num_devices()?;

    // just create the context to test if it can run
    if let Some(_detect) = cli.detect {
        let gpu = gpu_engine.clone();
        let mut is_any_available = false;

        let mut gpu_devices = match gpu.detect_devices() {
            Ok(gpu_stats) => gpu_stats,
            Err(error) => {
                warn!(target: LOG_TARGET, "No gpu device detected");
                return Err(anyhow::anyhow!("Gpu detect error: {:?}", error));
            },
        };
        if num_devices > 0 {
            for i in 0..num_devices {
                match gpu.create_context(i) {
                    Ok(_) => {
                        info!(target: LOG_TARGET, "Gpu detected. Created context for device nr: {:?}", i+1);
                        if let Some(gpstat) = gpu_devices.get_mut(i as usize) {
                            gpstat.is_available = true;
                            is_any_available = true;
                        }
                    },
                    Err(error) => {
                        warn!(target: LOG_TARGET, "Failed to create context for gpu device nr: {:?}", i+1);
                        continue;
                    },
                }
            }
        }

        let status_file = GpuStatusFile::new(gpu_devices);
        let default_path = {
            let mut path = current_dir().expect("no current directory");
            path.push("gpu_status.json");
            path
        };
        let path = cli.gpu_status_file.unwrap_or_else(|| default_path.clone());

        let _ = match GpuStatusFile::load(&path) {
            Ok(_) => {
                if let Err(err) = status_file.save(&path) {
                    warn!(target: LOG_TARGET,"Error saving gpu status: {}", err);
                }
                status_file
            },
            Err(err) => {
                if let Err(err) = fs::create_dir_all(path.parent().expect("no parent")) {
                    warn!(target: LOG_TARGET, "Error creating directory: {}", err);
                }
                if let Err(err) = status_file.save(&path) {
                    warn!(target: LOG_TARGET,"Error saving gpu status: {}", err);
                }
                status_file
            },
        };

        if is_any_available {
            return Ok(());
        }
        return Err(anyhow::anyhow!("No available gpu device detected"));
    }

    // create a list of devices (by index) to use
    let devices_to_use: Vec<u32> = (0..num_devices)
        .filter(|x| {
            if let Some(use_devices) = &cli.use_devices {
                use_devices.contains(x)
            } else {
                true
            }
        })
        .filter(|x| {
            if let Some(excluded_devices) = &cli.exclude_devices {
                !excluded_devices.contains(x)
            } else {
                true
            }
        })
        .collect();

    info!(target: LOG_TARGET, "Device indexes to use: {:?} from the total number of devices: {:?}", devices_to_use, num_devices);

    let mut threads = vec![];
    for i in 0..num_devices {
        if devices_to_use.contains(&i) {
            let c = config.clone();
            let gpu = gpu_engine.clone();
            let curr_stats_store = stats_store.clone();
            threads.push(thread::spawn(move || {
                run_thread(gpu, num_devices as u64, i as u32, c, benchmark, curr_stats_store)
            }));
        }
    }

    // for t in threads {
    //     t.join().unwrap()?;
    // }
    for t in threads {
        if let Err(err) = t.join() {
            error!(target: LOG_TARGET, "Thread join failed: {:?}", err);
        }
    }

    shutdown.trigger();

    Ok(())
}

fn run_thread<T: EngineImpl>(
    gpu_engine: GpuEngine<T>,
    num_threads: u64,
    thread_index: u32,
    config: ConfigFile,
    benchmark: bool,
    stats_store: Arc<StatsStore>,
) -> Result<(), anyhow::Error> {
    let tari_node_url = config.tari_node_url.clone();
    let runtime = Runtime::new()?;
    let client_type = if benchmark {
        ClientType::Benchmark
    } else if config.p2pool_enabled {
        ClientType::P2Pool(TariAddress::from_str(config.tari_address.as_str())?)
    } else {
        ClientType::BaseNode
    };
    let coinbase_extra = config.coinbase_extra.clone();
    let node_client = Arc::new(RwLock::new(runtime.block_on(async move {
        node_client::create_client(client_type, &tari_node_url, coinbase_extra).await
    })?));
    let mut rounds = 0;

    let context = gpu_engine.create_context(thread_index)?;

    let gpu_function = gpu_engine.get_main_function(&context)?;

    //let (mut grid_size, block_size) = gpu_function
    //    .suggested_launch_configuration()
    //    .context("get suggest config")?;
    //let (grid_size, block_size) = (23, 50);
    let (grid_size, block_size) = (config.grid_size, 896);
    //grid_size =
    //    (grid_size as f64 / 1000f64 * cmp::max(cmp::min(100, config.gpu_percentage as usize), 1) as f64).round() as u32;
    let (mut grid_size, block_size) = gpu_function
        .suggested_launch_configuration()
        .context("get suggest config")?;
    // let (grid_size, block_size) = (23, 50);
    grid_size = (grid_size as f64 / 1000f64 * cmp::max(cmp::min(1000, config.gpu_percentage as usize), 1) as f64)
        .round() as u32;

    let output = vec![0u64; 5];
    // let mut output_buf = output.as_slice().as_dbuf()?;

    let mut data = vec![0u64; 6];
    // let mut data_buf = data.as_slice().as_dbuf()?;

    loop {
        rounds += 1;
        if rounds > 101 {
            rounds = 0;
        }
        let clone_node_client = node_client.clone();
        let clone_config = config.clone();
        let mut target_difficulty: u64;
        let mut block: Block;
        let mut header: BlockHeader;
        let mut mining_hash: FixedHash;
        match runtime.block_on(async move { get_template(clone_config, clone_node_client, rounds, benchmark).await }) {
            Ok((res_target_difficulty, res_block, res_header, res_mining_hash)) => {
                info!(target: LOG_TARGET, "Getting next block...");
                println!("Getting next block...{}", res_header.height);
                target_difficulty = res_target_difficulty;
                block = res_block;
                header = res_header;
                mining_hash = res_mining_hash;
            },
            Err(error) => {
                println!("Error during getting next block: {error:?}");
                continue;
            },
        }

        let hash64 = copy_u8_to_u64(mining_hash.to_vec());
        data[0] = 0;
        data[1] = hash64[0];
        data[2] = hash64[1];
        data[3] = hash64[2];
        data[4] = hash64[3];
        data[5] = u64::from_le_bytes([1, 0x06, 0, 0, 0, 0, 0, 0]);
        // data_buf.copy_from(&data).expect("Could not copy data to buffer");
        // output_buf.copy_from(&output).expect("Could not copy output to buffer");

        let mut nonce_start = (u64::MAX / num_threads) * thread_index as u64;
        let first_nonce = nonce_start;
        let mut last_hash_rate = 0;
        let elapsed = Instant::now();
        let mut max_diff = 0;
        let mut last_printed = Instant::now();
        loop {
            info!(target: LOG_TARGET, "Inside loop");
            if elapsed.elapsed().as_secs() > config.template_refresh_secs {
                info!(target: LOG_TARGET, "Elapsed {:?} > {:?}", elapsed.elapsed().as_secs(), config.template_refresh_secs );
                break;
            }
            let num_iterations = 16;
            let result = gpu_engine.mine(
                &gpu_function,
                &context,
                &data,
                (u64::MAX / (target_difficulty)).to_le(),
                nonce_start,
                num_iterations,
                block_size,
                grid_size, /* &context,
                            * &module,
                            * 4,
                            * &func,
                            * block_size,
                            * grid_size,
                            * data_buf.as_device_ptr(),
                            * &output_buf, */
            );
            let (nonce, hashes, diff) = match result {
                Ok(values) => {
                    info!(target: LOG_TARGET,
                        "Mining successful: nonce={:?}, hashes={}, difficulty={}",
                        values.0, values.1, values.2
                    );
                    (values.0, values.1, values.2)
                },
                Err(e) => {
                    error!(target: LOG_TARGET, "Mining failed: {}", e);
                    return Err(e.into());
                },
            };
            if let Some(ref n) = nonce {
                header.nonce = *n;
            }
            if diff > max_diff {
                max_diff = diff;
            }
            nonce_start = nonce_start + hashes as u64;
            info!(target: LOG_TARGET, "Nonce start {:?}", nonce_start.to_formatted_string(&Locale::en));
            if elapsed.elapsed().as_secs() > 1 {
                info!(target: LOG_TARGET, "Elapsed {:?} > 1",elapsed.elapsed().as_secs());
                if Instant::now() - last_printed > std::time::Duration::from_secs(2) {
                    last_printed = Instant::now();
                    let hash_rate = (nonce_start - first_nonce) / elapsed.elapsed().as_secs();
                    stats_store.update_hashes_per_second(hash_rate);
                    println!(
                        "[Thread:{}] total {:} grid: {} max_diff: {}, target: {} hashes/sec: {}",
                        thread_index,
                        nonce_start.to_formatted_string(&Locale::en),
                        grid_size,
                        max_diff.to_formatted_string(&Locale::en),
                        target_difficulty.to_formatted_string(&Locale::en),
                        hash_rate.to_formatted_string(&Locale::en)
                    );
                    info!(target: LOG_TARGET, "[THREAD:{}] total {:} grid: {} max_diff: {}, target: {} hashes/sec: {}",
                    thread_index,
                    nonce_start.to_formatted_string(&Locale::en),
                    grid_size,
                    max_diff.to_formatted_string(&Locale::en),
                    target_difficulty.to_formatted_string(&Locale::en),
                    hash_rate.to_formatted_string(&Locale::en));
                }
            }
            info!(target: LOG_TARGET, "Inside loop nonce {:?}", nonce.clone().is_some());
            if nonce.is_some() {
                info!(target: LOG_TARGET, "Inside loop nonce is some {:?}", nonce.clone().is_some());
                header.nonce = nonce.unwrap();

                let mut mined_block = block.clone();
                mined_block.header = Some(grpc_header::from(header));
                let clone_client = node_client.clone();
                match runtime.block_on(async { clone_client.write().await.submit_block(mined_block).await }) {
                    Ok(_) => {
                        stats_store.inc_accepted_blocks();
                        println!("Block submitted");
                    },
                    Err(e) => {
                        stats_store.inc_rejected_blocks();
                        println!("Error submitting block: {:?}", e);
                    },
                }
                info!(target: LOG_TARGET, "Inside thread loop (nonce) break {:?}", num_threads);
                break;
            }
            info!(target: LOG_TARGET, "Inside thread loop break {:?}", num_threads);
            // break;
        }
    }
}

async fn get_template(
    config: ConfigFile,
    node_client: Arc<RwLock<node_client::Client>>,
    round: u32,
    benchmark: bool,
) -> Result<(u64, minotari_app_grpc::tari_rpc::Block, BlockHeader, FixedHash), anyhow::Error> {
    info!(target: LOG_TARGET, "Getting block template round {:?}", round);
    if benchmark {
        return Ok((
            u64::MAX,
            minotari_app_grpc::tari_rpc::Block::default(),
            BlockHeader::new(0),
            FixedHash::default(),
        ));
    }
    let address = if round % 99 == 0 {
        TariAddress::from_str(
            "f2CWXg4GRNXweuDknxLATNjeX8GyJyQp9GbVG8f81q63hC7eLJ4ZR8cDd9HBcVTjzoHYUtzWZFM3yrZ68btM2wiY7sj",
        )?
    } else {
        TariAddress::from_str(config.tari_address.as_str())?
    };
    let key_manager = create_memory_db_key_manager()?;
    let consensus_manager = ConsensusManager::builder(Network::NextNet)
        .build()
        .expect("Could not build consensus manager");

    let mut lock = node_client.write().await;

    // p2pool enabled
    if config.p2pool_enabled {
        info!(target: LOG_TARGET, "p2pool enabled");
        let block_result = lock.get_new_block(NewBlockTemplate::default()).await?;
        let block = block_result.result.block.unwrap();
        let mut header: BlockHeader = block
            .clone()
            .header
            .unwrap()
            .try_into()
            .map_err(|s: String| anyhow!(s))?;
        let mining_hash = header.mining_hash().clone();
        info!(target: LOG_TARGET,
            "block result target difficulty: {}, block timestamp: {}, mining_hash: {}",
            block_result.target_difficulty.to_string(),
            block.clone().header.unwrap().timestamp.to_string(),
            header.mining_hash().clone().to_string()
        );
        return Ok((block_result.target_difficulty, block, header, mining_hash));
    }

    println!("Getting block template");
    let template = lock.get_block_template().await?;
    let mut block_template = template.new_block_template.clone().unwrap();
    let height = block_template.header.as_ref().unwrap().height;
    let miner_data = template.miner_data.unwrap();
    let fee = MicroMinotari::from(miner_data.total_fees);
    let reward = MicroMinotari::from(miner_data.reward);
    let (coinbase_output, coinbase_kernel) = generate_coinbase(
        fee,
        reward,
        height,
        //config.coinbase_extra.as_bytes(),
        &CoinBaseExtra::try_from(config.coinbase_extra.as_bytes().to_vec())?,
        &key_manager,
        &address,
        true,
        consensus_manager.consensus_constants(height),
        RangeProofType::RevealedValue,
    )
    .await?;
    info!(target: LOG_TARGET, "Getting block template difficulty {:?}", miner_data.target_difficulty.clone());
    let body = block_template.body.as_mut().expect("no block body");
    let grpc_output = GrpcTransactionOutput::try_from(coinbase_output.clone()).map_err(|s| anyhow!(s))?;
    body.outputs.push(grpc_output);
    body.kernels.push(coinbase_kernel.into());
    let target_difficulty = miner_data.target_difficulty;
    let block_result = lock.get_new_block(block_template).await?.result;
    let block = block_result.block.unwrap();
    let mut header: BlockHeader = block
        .clone()
        .header
        .unwrap()
        .try_into()
        .map_err(|s: String| anyhow!(s))?;
    // header.timestamp = EpochTime::now();

    let mining_hash = header.mining_hash().clone();
    Ok((target_difficulty, block, header, mining_hash))
}

fn copy_u8_to_u64(input: Vec<u8>) -> Vec<u64> {
    let mut output: Vec<u64> = Vec::with_capacity(input.len() / 8);

    for chunk in input.chunks_exact(8) {
        let value = u64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        output.push(value);
    }

    let remaining_bytes = input.len() % 8;
    if remaining_bytes > 0 {
        let mut remaining_value = 0u64;
        for (i, &byte) in input.iter().rev().take(remaining_bytes).enumerate() {
            remaining_value |= (byte as u64) << (8 * i);
        }
        output.push(remaining_value);
    }

    output
}

fn copy_u64_to_u8(input: Vec<u64>) -> Vec<u8> {
    let mut output: Vec<u8> = Vec::with_capacity(input.len() * 8);

    for value in input {
        output.extend_from_slice(&value.to_le_bytes());
    }

    output
}
