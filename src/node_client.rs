use std::time::{Duration, Instant};

use anyhow::anyhow;
use log::{error, info, warn};
use minotari_app_grpc::tari_rpc::{
    base_node_client::BaseNodeClient,
    pow_algo::PowAlgos,
    sha_p2_pool_client::ShaP2PoolClient,
    Block,
    Empty,
    GetNewBlockResult,
    NewBlockTemplate,
    NewBlockTemplateRequest,
    NewBlockTemplateResponse,
    PowAlgo,
};
use tari_common_types::{tari_address::TariAddress, types::FixedHash};
use tonic::{async_trait, transport::Channel};

use crate::p2pool_client::P2poolClientWrapper;

const LOG_TARGET: &str = "tari::gpuminer::node-client";

pub(crate) struct BaseNodeClientWrapper {
    client: BaseNodeClient<tonic::transport::Channel>,
}

impl BaseNodeClientWrapper {
    pub async fn connect(url: &str) -> Result<Self, anyhow::Error> {
        println!("Connecting to {}", url);
        info!(target: LOG_TARGET, "Connecting to {}", url);
        let mut client: Option<BaseNodeClient<Channel>> = None;
        while client.is_none() {
            match BaseNodeClient::connect(url.to_string()).await {
                Ok(res_client) => {
                    info!(target: LOG_TARGET, "Connected successfully");
                    client = Some(res_client)
                },
                Err(error) => {
                    error!(target: LOG_TARGET,"Failed to connect to base node: {:?}", error);
                    println!("Failed to connect to base node: {error:?}");
                    tokio::time::sleep(Duration::from_secs(5)).await;
                },
            }
        }

        Ok(Self {
            client: client.unwrap(),
        })
    }
}

#[async_trait]
impl NodeClient for BaseNodeClientWrapper {
    async fn get_version(&mut self) -> Result<u64, anyhow::Error> {
        info!(target: LOG_TARGET, "Getting node client version");
        let res = self.client.get_version(tonic::Request::new(Empty {})).await?;
        // dbg!(res);
        Ok(0)
    }

    async fn get_block_template(&mut self) -> Result<NewBlockTemplateResponse, anyhow::Error> {
        info!(target: LOG_TARGET, "Getting node block template");
        let res = self
            .client
            .get_new_block_template(tonic::Request::new({
                NewBlockTemplateRequest {
                    max_weight: 0,
                    algo: Some(PowAlgo {
                        pow_algo: PowAlgos::Sha3x.into(),
                    }),
                }
            }))
            .await?;
        info!(target: LOG_TARGET, "Done getting node block template");
        Ok(res.into_inner())
    }

    async fn get_new_block(&mut self, template: NewBlockTemplate) -> Result<NewBlockResult, anyhow::Error> {
        info!(target: LOG_TARGET, "Getting new block template");
        let res = self.client.get_new_block(tonic::Request::new(template)).await?;
        Ok(NewBlockResult::try_from(res.into_inner())?)
    }

    async fn submit_block(&mut self, block: Block) -> Result<(), anyhow::Error> {
        info!(target: LOG_TARGET, "Submitting block");
        // dbg!(&block);
        let res = self.client.submit_block(tonic::Request::new(block)).await?;
        info!(target: LOG_TARGET, "Block submitted: {:?}", res);
        Ok(())
    }

    async fn get_height(&mut self) -> Result<HeightData, anyhow::Error> {
        todo!()
    }
}

#[async_trait]
pub trait NodeClient {
    async fn get_version(&mut self) -> Result<u64, anyhow::Error>;

    async fn get_height(&mut self) -> Result<HeightData, anyhow::Error>;

    async fn get_block_template(&mut self) -> Result<NewBlockTemplateResponse, anyhow::Error>;

    async fn get_new_block(&mut self, template: NewBlockTemplate) -> Result<NewBlockResult, anyhow::Error>;

    async fn submit_block(&mut self, block: Block) -> Result<(), anyhow::Error>;
}

pub(crate) async fn create_client(
    client_type: ClientType,
    url: &str,
    coinbase_extra: String,
) -> Result<Client, anyhow::Error> {
    info!(target: LOG_TARGET, "Creating node client: {}", url);
    Ok(match client_type {
        ClientType::BaseNode => Client::BaseNode(BaseNodeClientWrapper::connect(url).await?),
        ClientType::Benchmark => Client::Benchmark(BenchmarkNodeClient {}),
        ClientType::P2Pool(wallet_payment_address) => {
            Client::P2Pool(P2poolClientWrapper::connect(url, wallet_payment_address, coinbase_extra).await?)
        },
    })
}

pub(crate) enum Client {
    BaseNode(BaseNodeClientWrapper),
    P2Pool(P2poolClientWrapper),
    Benchmark(BenchmarkNodeClient),
}

pub enum ClientType {
    BaseNode,
    Benchmark,
    P2Pool(TariAddress),
}

pub struct NewBlockResult {
    pub result: GetNewBlockResult,
    pub target_difficulty: u64,
}

impl TryFrom<GetNewBlockResult> for NewBlockResult {
    type Error = anyhow::Error;

    fn try_from(result: GetNewBlockResult) -> Result<Self, Self::Error> {
        let target_difficulty = result
            .miner_data
            .clone()
            .ok_or(anyhow!("missing miner data"))?
            .target_difficulty;
        Ok(Self {
            result,
            target_difficulty,
        })
    }
}

pub(crate) struct HeightData {
    pub height: u64,
    pub tip_hash: FixedHash,
    pub p2pool_height: u64,
    pub p2pool_tip_hash: FixedHash,
}

impl Client {
    pub async fn get_version(&mut self) -> Result<u64, anyhow::Error> {
        match self {
            Client::BaseNode(client) => client.get_version().await,
            Client::Benchmark(client) => client.get_version().await,
            Client::P2Pool(client) => client.get_version().await,
        }
    }

    pub async fn get_height(&mut self) -> Result<HeightData, anyhow::Error> {
        match self {
            Client::BaseNode(client) => client.get_height().await,
            Client::Benchmark(client) => client.get_height().await,
            Client::P2Pool(client) => client.get_height().await,
        }
    }

    pub async fn get_block_template(&mut self) -> Result<NewBlockTemplateResponse, anyhow::Error> {
        let timer = Instant::now();
        match self {
            Client::BaseNode(client) => client.get_block_template().await,
            Client::Benchmark(client) => client.get_block_template().await,
            Client::P2Pool(client) => client.get_block_template().await,
        }
        .map(|res| {
            if timer.elapsed() > Duration::from_secs(5) {
                warn!(target: LOG_TARGET, "⚠ SLOW GET_BLOCK_TEMPLATE: Get_block_template took {:?}. Target is 5 seconds", timer.elapsed());
            }
            res
        })
    }

    pub async fn get_new_block(&mut self, template: NewBlockTemplate) -> Result<NewBlockResult, anyhow::Error> {
        let timer = Instant::now();
        match self {
            Client::BaseNode(client) => client.get_new_block(template).await,
            Client::Benchmark(client) => client.get_new_block(template).await,
            Client::P2Pool(client) => client.get_new_block(template).await,
        }
        .map(|res| {
            if timer.elapsed() > Duration::from_secs(5) {
                warn!(target: LOG_TARGET, "⚠ SLOW GET_NEW_BLOCK: Get_new_block took {:?}. Target is 5 seconds", timer.elapsed());
            }
            res
        })
    }

    pub async fn submit_block(&mut self, block: Block) -> Result<(), anyhow::Error> {
        let timer = Instant::now();
        let res = match self {
            Client::BaseNode(client) => client.submit_block(block).await,
            Client::Benchmark(client) => client.submit_block(block).await,
            Client::P2Pool(client) => client.submit_block(block).await,
        };
        if timer.elapsed() > Duration::from_secs(5) {
            error!(target: LOG_TARGET, "⚠ SLOW SUBMIT: Submit_block took {:?}. Target is 5 seconds", timer.elapsed());
        }
        res
    }
}

pub(crate) struct BenchmarkNodeClient {}

#[async_trait]
impl NodeClient for BenchmarkNodeClient {
    async fn get_version(&mut self) -> Result<u64, anyhow::Error> {
        Ok(0)
    }

    async fn get_block_template(&mut self) -> Result<NewBlockTemplateResponse, anyhow::Error> {
        todo!()
    }

    async fn get_new_block(&mut self, template: NewBlockTemplate) -> Result<NewBlockResult, anyhow::Error> {
        todo!()
    }

    async fn submit_block(&mut self, block: Block) -> Result<(), anyhow::Error> {
        Ok(())
    }

    async fn get_height(&mut self) -> Result<HeightData, anyhow::Error> {
        todo!()
    }
}
