use std::sync::Arc;

use axum::{routing::get, Router};
use tari_shutdown::ShutdownSignal;
use thiserror::Error;
use tokio::io;

use crate::{
    http::{
        config,
        handlers::{health, stats, version},
    },
    stats_store::StatsStore,
};

use log::info;
const LOG_TARGET: &str = "tari::gpuminer::server";

/// An HTTP server that provides stats and other useful information.
pub struct HttpServer {
    shutdown_signal: ShutdownSignal,
    config: config::Config,
    stats_store: Arc<StatsStore>,
}

#[derive(Error, Debug)]
pub enum Error {
    #[error("I/O error: {0}")]
    IO(#[from] io::Error),
}

#[derive(Clone)]
pub struct AppState {
    pub stats_store: Arc<StatsStore>,
}

impl HttpServer {
    pub fn new(shutdown_signal: ShutdownSignal, config: config::Config, stats_store: Arc<StatsStore>) -> Self {
        Self {
            shutdown_signal,
            config,
            stats_store,
        }
    }

    pub fn routes(&self) -> Router {
        Router::new()
            .route("/health", get(health::handle_health))
            .route("/version", get(version::handle_version))
            .route("/stats", get(stats::handle_get_stats))
            .with_state(AppState {
                stats_store: self.stats_store.clone(),
            })
    }

    /// Starts the http server on the port passed in ['HttpServer::new']
    pub async fn start(&self) -> Result<(), Error> {
        let router = self.routes();
        let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", self.config.port))
            .await
            .map_err(Error::IO)?;
        println!("Starting HTTP server at http://127.0.0.1:{}", self.config.port);
        println!("Starting HTTP listener address {:?}", listener.local_addr());
        info!(target: LOG_TARGET, "Starting HTTP listener address {:?}", listener.local_addr());
        axum::serve(listener, router)
            .with_graceful_shutdown(self.shutdown_signal.clone())
            .await
            .map_err(Error::IO)?;
        println!("HTTP server stopped!");
        Ok(())
    }
}
