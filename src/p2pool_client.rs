use anyhow::{anyhow, Error};
use minotari_app_grpc::tari_rpc::sha_p2_pool_client::ShaP2PoolClient;
use minotari_app_grpc::tari_rpc::{
    Block, GetNewBlockRequest, NewBlockTemplate, NewBlockTemplateResponse, SubmitBlockRequest,
};
use std::time::Duration;
use tari_common_types::tari_address::TariAddress;
use tonic::async_trait;
use tonic::transport::Channel;

use crate::node_client::{NewBlockResult, NodeClient};

pub struct P2poolClientWrapper {
    client: ShaP2PoolClient<Channel>,
    wallet_payment_address: TariAddress,
}

impl P2poolClientWrapper {
    pub async fn connect(url: &str, wallet_payment_address: TariAddress) -> Result<Self, anyhow::Error> {
        println!("Connecting to {}", url);
        let mut client: Option<ShaP2PoolClient<Channel>> = None;
        while client.is_none() {
            match ShaP2PoolClient::connect(url.to_string()).await {
                Ok(res_client) => client = Some(res_client),
                Err(error) => {
                    println!("Failed to connect to p2pool node: {error:?}");
                    tokio::time::sleep(Duration::from_secs(5)).await;
                },
            }
        }

        Ok(Self {
            client: client.unwrap(),
            wallet_payment_address,
        })
    }
}

#[async_trait]
impl NodeClient for P2poolClientWrapper {
    async fn get_version(&mut self) -> Result<u64, Error> {
        Ok(0)
    }

    async fn get_block_template(&mut self) -> Result<NewBlockTemplateResponse, Error> {
        Err(anyhow!("not supported"))
    }

    async fn get_new_block(&mut self, _template: NewBlockTemplate) -> Result<NewBlockResult, Error> {
        let response = self
            .client
            .get_new_block(GetNewBlockRequest::default())
            .await?
            .into_inner();
        Ok(NewBlockResult {
            result: response.block.ok_or(anyhow!("missing block response"))?,
            target_difficulty: response.target_difficulty,
        })
    }

    async fn submit_block(&mut self, block: Block) -> Result<(), Error> {
        self.client
            .submit_block(SubmitBlockRequest {
                block: Some(block),
                wallet_payment_address: self.wallet_payment_address.to_base58(),
            })
            .await?;
        Ok(())
    }
}
