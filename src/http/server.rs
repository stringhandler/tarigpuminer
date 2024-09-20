use crate::http::config;
use crate::http::handlers::{health, stats, version};
use crate::stats_store::StatsStore;
use axum::routing::get;
use axum::Router;
use log::{error, info};
use std::sync::Arc;
use tari_shutdown::ShutdownSignal;
use thiserror::Error;
use tokio::io;
const LOG_TARGET: &str = "tari::gpuminer::httpserver";

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
        info!(target: LOG_TARGET, "Http: starts the http server on the port {:?}", self.config.port);
        let router = self.routes();
        let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", self.config.port))
            .await
            .map_err(Error::IO)?;
        println!("Starting HTTP server at http://127.0.0.1:{}", self.config.port);
        info!(target: LOG_TARGET, "Http: http server is running at http://127.0.0.1:{}", self.config.port);
        axum::serve(listener, router)
            .with_graceful_shutdown(self.shutdown_signal.clone())
            .await
            .map_err(Error::IO)?;
        println!("HTTP server stopped!");
        info!(target: LOG_TARGET, "HTTP server stopped!");
        Ok(())
    }
}
