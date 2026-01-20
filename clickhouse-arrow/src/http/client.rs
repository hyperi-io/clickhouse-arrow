//! HTTP client for `ClickHouse`.

use arrow::array::RecordBatch;
use bytes::Bytes;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use tracing::{debug, instrument, trace_span, Instrument};

use super::arrow_stream::{deserialize_batches, serialize_batch};
use super::config::HttpOptions;
use crate::errors::Result;
use crate::Error;

/// HTTP client for `ClickHouse` using `ArrowStream` format.
///
/// This client provides an alternative to the native TCP protocol, using HTTP
/// with `ClickHouse`'s `FORMAT ArrowStream` for efficient Arrow data exchange.
///
/// # Example
///
/// ```rust,ignore
/// use clickhouse_arrow::http::{HttpClient, HttpOptions};
///
/// let options = HttpOptions::new("http://localhost:8123")?
///     .with_database("default")
///     .with_credentials("default", "");
///
/// let client = HttpClient::new(options)?;
///
/// // Execute a query
/// let batches = client.query("SELECT * FROM my_table").await?;
///
/// // Insert data
/// client.insert("my_table", batch).await?;
/// ```
#[derive(Debug, Clone)]
pub struct HttpClient {
    client:  reqwest::Client,
    options: HttpOptions,
}

impl HttpClient {
    /// Create a new HTTP client with the given options.
    ///
    /// # Errors
    /// Returns an error if the reqwest client cannot be built.
    pub fn new(options: HttpOptions) -> Result<Self> {
        let mut builder = reqwest::Client::builder()
            .timeout(options.timeout)
            .use_rustls_tls();

        if options.enable_compression {
            builder = builder.gzip(true).zstd(true);
        }

        let client = builder
            .build()
            .map_err(|e| Error::Configuration(format!("Failed to build HTTP client: {e}")))?;

        Ok(Self { client, options })
    }

    /// Build default headers for requests.
    fn default_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();

        if let Some(ref user) = self.options.user
            && let Ok(value) = HeaderValue::from_str(user) {
                drop(headers.insert("X-ClickHouse-User", value));
            }

        if let Some(ref password) = self.options.password
            && let Ok(value) = HeaderValue::from_str(password) {
                drop(headers.insert("X-ClickHouse-Key", value));
            }

        if let Some(ref database) = self.options.database
            && let Ok(value) = HeaderValue::from_str(database) {
                drop(headers.insert("X-ClickHouse-Database", value));
            }

        headers
    }

    /// Build the query URL with the given SQL and format.
    fn build_query_url(&self, sql: &str, format: &str) -> url::Url {
        let mut url = self.options.url.clone();

        // Append FORMAT to the query
        let query_with_format = format!("{sql} FORMAT {format}");

        let _ = url.query_pairs_mut().append_pair("query", &query_with_format);

        url
    }

    /// Execute a SELECT query and return results as Arrow `RecordBatch`es.
    ///
    /// The query is executed with `FORMAT ArrowStream` and the response is
    /// deserialized into Arrow record batches.
    ///
    /// # Errors
    /// Returns an error if:
    /// - The HTTP request fails
    /// - The response indicates an error
    /// - The `ArrowStream` cannot be deserialized
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let batches = client.query("SELECT id, name FROM users WHERE active = 1").await?;
    /// for batch in batches {
    ///     println!("Got {} rows", batch.num_rows());
    /// }
    /// ```
    #[must_use = "query results should be used"]
    #[instrument(skip(self), fields(sql = %sql))]
    pub async fn query(&self, sql: &str) -> Result<Vec<RecordBatch>> {
        let url = self.build_query_url(sql, "ArrowStream");
        let headers = self.default_headers();

        debug!(url = %url, "Executing HTTP query");

        let response = self
            .client
            .get(url)
            .headers(headers)
            .send()
            .instrument(trace_span!("http_request"))
            .await
            .map_err(|e| Error::Network(e.to_string()))?;

        self.handle_response(response).await
    }

    /// Execute a DDL or other non-returning query.
    ///
    /// Use this for CREATE, DROP, ALTER, and other statements that don't return data.
    ///
    /// # Errors
    /// Returns an error if:
    /// - The HTTP request fails
    /// - The response indicates an error
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// client.execute("CREATE TABLE IF NOT EXISTS users (id UInt64, name String) ENGINE = MergeTree ORDER BY id").await?;
    /// ```
    #[instrument(skip(self), fields(sql = %sql))]
    pub async fn execute(&self, sql: &str) -> Result<()> {
        let mut url = self.options.url.clone();
        let _ = url.query_pairs_mut().append_pair("query", sql);

        let headers = self.default_headers();

        debug!(url = %url, "Executing HTTP DDL");

        let response = self
            .client
            .post(url)
            .headers(headers)
            .send()
            .instrument(trace_span!("http_request"))
            .await
            .map_err(|e| Error::Network(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Server(format!("HTTP {status}: {body}")));
        }

        Ok(())
    }

    /// Insert Arrow data into a table.
    ///
    /// The `RecordBatch` is serialized to `ArrowStream` format and sent to `ClickHouse`.
    ///
    /// # Errors
    /// Returns an error if:
    /// - The batch cannot be serialized
    /// - The HTTP request fails
    /// - The response indicates an error
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let batch = create_record_batch();
    /// client.insert("users", batch).await?;
    /// ```
    #[instrument(skip(self, batch), fields(table = %table, rows = batch.num_rows()))]
    pub async fn insert(&self, table: &str, batch: RecordBatch) -> Result<()> {
        let sql = format!("INSERT INTO {table} FORMAT ArrowStream");
        let mut url = self.options.url.clone();
        let _ = url.query_pairs_mut().append_pair("query", &sql);

        let mut headers = self.default_headers();
        drop(headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/octet-stream")));

        let body = serialize_batch(&batch)?;

        debug!(url = %url, body_size = body.len(), "Executing HTTP insert");

        let response = self
            .client
            .post(url)
            .headers(headers)
            .body(body)
            .send()
            .instrument(trace_span!("http_request"))
            .await
            .map_err(|e| Error::Network(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Server(format!("HTTP {status}: {body}")));
        }

        Ok(())
    }

    /// Insert multiple Arrow batches into a table.
    ///
    /// All batches must have the same schema. They are combined into a single
    /// `ArrowStream` and sent in one request.
    ///
    /// # Errors
    /// Returns an error if:
    /// - The batches have mismatched schemas
    /// - Serialization fails
    /// - The HTTP request fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let batches = vec![batch1, batch2, batch3];
    /// client.insert_batches("users", batches).await?;
    /// ```
    #[instrument(skip(self, batches), fields(table = %table, batch_count = batches.len()))]
    pub async fn insert_batches(&self, table: &str, batches: Vec<RecordBatch>) -> Result<()> {
        if batches.is_empty() {
            return Ok(());
        }

        let sql = format!("INSERT INTO {table} FORMAT ArrowStream");
        let mut url = self.options.url.clone();
        let _ = url.query_pairs_mut().append_pair("query", &sql);

        let mut headers = self.default_headers();
        drop(headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/octet-stream")));

        let body = serialize_batches(&batches)?;

        debug!(url = %url, body_size = body.len(), "Executing HTTP batch insert");

        let response = self
            .client
            .post(url)
            .headers(headers)
            .body(body)
            .send()
            .instrument(trace_span!("http_request"))
            .await
            .map_err(|e| Error::Network(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Server(format!("HTTP {status}: {body}")));
        }

        Ok(())
    }

    /// Handle an HTTP response, checking for errors and deserializing `ArrowStream`.
    async fn handle_response(&self, response: reqwest::Response) -> Result<Vec<RecordBatch>> {
        let status = response.status();

        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Server(format!("HTTP {status}: {body}")));
        }

        let body = response
            .bytes()
            .instrument(trace_span!("read_response"))
            .await
            .map_err(|e| Error::Network(format!("Failed to read response body: {e}")))?;

        deserialize_batches(body)
    }
}

/// Serialize multiple batches to `ArrowStream` format.
fn serialize_batches(batches: &[RecordBatch]) -> Result<Bytes> {
    use arrow::ipc::writer::StreamWriter;

    if batches.is_empty() {
        return Ok(Bytes::new());
    }

    let schema = batches[0].schema();
    let total_size: usize = batches.iter().map(RecordBatch::get_array_memory_size).sum();
    let mut buffer = Vec::with_capacity(total_size);

    let mut writer = StreamWriter::try_new(&mut buffer, &schema)
        .map_err(|e| Error::ArrowSerialize(format!("Failed to create ArrowStream writer: {e}")))?;

    for batch in batches {
        writer
            .write(batch)
            .map_err(|e| Error::ArrowSerialize(format!("Failed to write batch to ArrowStream: {e}")))?;
    }

    writer
        .finish()
        .map_err(|e| Error::ArrowSerialize(format!("Failed to finish ArrowStream: {e}")))?;

    Ok(Bytes::from(buffer))
}
