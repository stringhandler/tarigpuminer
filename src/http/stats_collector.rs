use std::{
    collections::{HashMap, VecDeque},
    thread::current,
};

use log::error;
use serde::Serialize;
use tari_shutdown::ShutdownSignal;
use tari_utilities::epoch_time::EpochTime;
use tokio::sync::{broadcast::Receiver, oneshot};

const LOG_TARGET: &str = "tari::gpu_miner::http::stats_collector";

pub(crate) struct StatsCollector {
    shutdown_signal: ShutdownSignal,
    hashrate_samples: HashMap<usize, VecDeque<HashrateSample>>,
    stats_broadcast_receiver: tokio::sync::broadcast::Receiver<HashrateSample>,
    request_tx: tokio::sync::mpsc::Sender<StatsRequest>,
    request_rx: tokio::sync::mpsc::Receiver<StatsRequest>,
}

pub(crate) enum StatsRequest {
    GetHashrate(tokio::sync::oneshot::Sender<GetHashrateResponse>),
}

pub(crate) struct GetHashrateResponse {
    pub devices: HashMap<usize, AverageHashrate>,
    pub total: AverageHashrate,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct AverageHashrate {
    ten_seconds: f64,
    one_minute: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct HashrateSample {
    pub(crate) device_id: usize,
    pub(crate) timestamp: EpochTime,
    pub(crate) hashrate: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct StatsClient {
    request_tx: tokio::sync::mpsc::Sender<StatsRequest>,
}

impl StatsClient {
    pub async fn get_hashrate(&self) -> Result<GetHashrateResponse, anyhow::Error> {
        let (tx, rx) = oneshot::channel();
        self.request_tx.send(StatsRequest::GetHashrate(tx)).await?;
        Ok(rx.await?)
    }
}
impl StatsCollector {
    pub(crate) fn new(shutdown_signal: ShutdownSignal, stats_broadcast_receiver: Receiver<HashrateSample>) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        Self {
            shutdown_signal,
            hashrate_samples: HashMap::new(),
            stats_broadcast_receiver,
            request_rx: rx,
            request_tx: tx,
        }
    }

    pub fn create_client(&self) -> StatsClient {
        StatsClient {
            request_tx: self.request_tx.clone(),
        }
    }

    fn calc_hashrate(&self) -> GetHashrateResponse {
        let mut result = HashMap::<usize, AverageHashrate>::new();
        let current_time = EpochTime::now().as_u64();
        for (device_id, s) in self.hashrate_samples.iter() {
            let mut samples: VecDeque<&HashrateSample> = s.iter().collect();
            let mut total_for_10_seconds = 0.0;
            let mut total_for_60_seconds = 0.0;
            for second in current_time - 60..current_time {
                // clear out anything older than 60 seconds
                loop {
                    if let Some(sample) = samples.front() {
                        if sample.timestamp.as_u64() < second {
                            samples.pop_front();
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }

                // only take the first sample we have for hashrate.
                if let Some(sample) = samples.front() {
                    if sample.timestamp.as_u64() == second {
                        total_for_60_seconds += sample.hashrate;
                        if second >= current_time - 10 {
                            total_for_10_seconds += sample.hashrate;
                        }
                        samples.pop_front();
                    }
                }
            }
            let ten_seconds = total_for_10_seconds / 10.0;
            let one_minute = total_for_60_seconds / 60.0;
            result.insert(*device_id, AverageHashrate {
                ten_seconds,
                one_minute,
            });
        }
        GetHashrateResponse {
            total: AverageHashrate {
                ten_seconds: result.values().map(|v| v.ten_seconds).sum(),
                one_minute: result.values().map(|v| v.one_minute).sum(),
            },
            devices: result,
        }
    }

    pub(crate) async fn run(&mut self) {
        loop {
            tokio::select! {
                _ = self.shutdown_signal.wait() => {
                    break;
                },
                res = self.request_rx.recv() => {
                    match res {
                        Some(StatsRequest::GetHashrate(tx)) => {
                            let hashrate = self.calc_hashrate();
                            let _ = tx.send(hashrate);
                        },
                        None => {
                            break;
                        }
                    }
                },
                res = self.stats_broadcast_receiver.recv() => {
                    match res {
                        Ok(sample) => {
                            let entry = self.hashrate_samples.entry(sample.device_id).or_insert_with(|| VecDeque::with_capacity(61));
                    if entry.len() > 60 {
                        entry.pop_front();
                    }
                    entry.push_back(sample);
                        },
                        Err(e) => {
                            error!(target: LOG_TARGET, "Error receiving hashrate sample: {:?}", e);
                            break;
                        }
                    }
                                    }
            }
        }
    }
}
