use std::str::FromStr;
use std::{convert::TryInto, env::current_dir, path::PathBuf, sync::Arc, thread, time::Instant};

use anyhow::{anyhow, Context as AnyContext, Error};
use clap::Parser;
#[cfg(feature = "nvidia")]
use cust::{
    memory::{AsyncCopyDestination, DeviceCopy},
    prelude::*,
};
use minotari_app_grpc::tari_rpc::{
    Block, BlockHeader as grpc_header, NewBlockTemplate, TransactionOutput as GrpcTransactionOutput,
};
use num_format::{Locale, ToFormattedString};
use sha3::Digest;
use tari_common::configuration::Network;
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
use crate::http::config::Config;
use crate::http::server::HttpServer;
use crate::node_client::ClientType;
#[cfg(feature = "opencl3")]
use crate::opencl_engine::OpenClEngine;
use crate::stats_store::StatsStore;
use crate::{
    config_file::ConfigFile, engine_impl::EngineImpl, function_impl::FunctionImpl, gpu_engine::GpuEngine,
    node_client::NodeClient, tari_coinbase::generate_coinbase,
};

mod config_file;
mod context_impl;
#[cfg(feature = "nvidia")]
mod cuda_engine;
mod engine_impl;
mod function_impl;
mod gpu_engine;
mod http;
mod node_client;
#[cfg(feature = "opencl3")]
mod opencl_engine;
mod p2pool_client;
mod stats_store;
mod tari_coinbase;

#[tokio::main]
async fn main() {
    match main_inner().await {
        Ok(()) => {},
        Err(err) => {
            eprintln!("Error: {:#?}", err);
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
}

async fn main_inner() -> Result<(), anyhow::Error> {
    let cli = Cli::parse();

    let benchmark = cli.benchmark;

    let mut config = match ConfigFile::load(cli.config.unwrap_or_else(|| {
        let mut path = current_dir().expect("no current directory");
        path.push("config.json");
        path
    })) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("Error loading config file: {}. Creating new one", err);
            let default = ConfigFile::default();
            default.save("config.json").expect("Could not save default config");
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
        let http_server = HttpServer::new(shutdown.to_signal(), http_server_config, stats_store.clone());
        tokio::spawn(async move {
            if let Err(error) = http_server.start().await {
                println!("Failed to start HTTP server: {error:?}");
            }
        });
    }

    let num_devices = gpu_engine.num_devices()?;
    let mut threads = vec![];
    for i in 0..num_devices {
        let c = config.clone();
        let gpu = gpu_engine.clone();
        let curr_stats_store = stats_store.clone();
        threads.push(thread::spawn(move || {
            run_thread(gpu, num_devices as u64, i as u32, c, benchmark, curr_stats_store)
        }));
    }

    for t in threads {
        t.join().unwrap()?;
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
    let node_client =
        Arc::new(RwLock::new(runtime.block_on(async move {
            node_client::create_client(client_type, &tari_node_url).await
        })?));
    let mut rounds = 0;

    let context = gpu_engine.create_context(thread_index)?;

    let gpu_function = gpu_engine.get_main_function(&context)?;

    let (grid_size, block_size) = gpu_function
        .suggested_launch_configuration()
        .context("get suggest config")?;
    // let (grid_size, block_size) = (23, 50);

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
        let mut last_hash_rate = 0;
        let elapsed = Instant::now();
        let mut max_diff = 0;
        let mut last_printed = Instant::now();
        loop {
            if elapsed.elapsed().as_secs() > config.template_refresh_secs {
                break;
            }
            let num_iterations = 16;
            let (nonce, hashes, diff) = gpu_engine.mine(
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
            )?;
            if let Some(ref n) = nonce {
                header.nonce = *n;
            }
            if diff > max_diff {
                max_diff = diff;
            }
            nonce_start = nonce_start + hashes as u64;
            if elapsed.elapsed().as_secs() > 1 {
                if Instant::now() - last_printed > std::time::Duration::from_secs(2) {
                    last_printed = Instant::now();
                    let hash_rate = nonce_start / elapsed.elapsed().as_secs();
                    stats_store.update_hashes_per_second(hash_rate);
                    println!(
                        "total {:} grid: {} max_diff: {}, target: {} hashes/sec: {}",
                        nonce_start.to_formatted_string(&Locale::en),
                        grid_size,
                        max_diff.to_formatted_string(&Locale::en),
                        target_difficulty.to_formatted_string(&Locale::en),
                        hash_rate.to_formatted_string(&Locale::en)
                    );
                }
            }
            if nonce.is_some() {
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
                break;
            }
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
        let block_result = lock.get_new_block(NewBlockTemplate::default()).await?;
        let block = block_result.result.block.unwrap();
        let mut header: BlockHeader = block
            .clone()
            .header
            .unwrap()
            .try_into()
            .map_err(|s: String| anyhow!(s))?;
        let mining_hash = header.mining_hash().clone();
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
        config.coinbase_extra.as_bytes(),
        &key_manager,
        &address,
        true,
        consensus_manager.consensus_constants(height),
        RangeProofType::RevealedValue,
    )
    .await?;
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
