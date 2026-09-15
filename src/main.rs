mod transcription;

use std::io::Write;

use actix_cors::Cors;
use actix_multipart::Multipart;
use actix_web::{App, HttpResponse, HttpServer, Responder, error, get, post, web};
use futures_util::TryStreamExt;
use serde::{Deserialize, Serialize};
use zip::{CompressionMethod, ZipWriter, write::SimpleFileOptions};

#[derive(Serialize)]
struct TranscriptionResponse {
    text: String,
}

#[derive(Deserialize)]
struct DownloadRequest {
    text: String,
    filename: Option<String>,
}

#[get("/health")]
async fn health() -> impl Responder {
    HttpResponse::Ok().json(serde_json::json!({ "status": "ok" }))
}

#[post("/transcribe")]
async fn transcribe(mut payload: Multipart) -> actix_web::Result<HttpResponse> {
    let mut audio = Vec::new();

    while let Some(mut field) = payload.try_next().await? {
        if field
            .content_disposition()
            .and_then(|cd| cd.get_name())
            .is_some_and(|name| name == "audio")
        {
            while let Some(chunk) = field.try_next().await? {
                audio.extend_from_slice(&chunk);
            }
            break;
        }
    }

    if audio.is_empty() {
        return Err(error::ErrorBadRequest(
            "missing multipart field: audio (wav expected)",
        ));
    }

    let text = web::block(move || transcription::transcribe_wav_bytes(&audio))
        .await
        .map_err(error::ErrorInternalServerError)?
        .map_err(error::ErrorBadRequest)?;

    Ok(HttpResponse::Ok().json(TranscriptionResponse { text }))
}

#[post("/download/txt")]
async fn download_txt(body: web::Json<DownloadRequest>) -> actix_web::Result<HttpResponse> {
    let filename = sanitize_filename(body.filename.as_deref().unwrap_or("transcription"));
    Ok(HttpResponse::Ok()
        .append_header((
            "Content-Disposition",
            format!("attachment; filename=\"{filename}.txt\""),
        ))
        .content_type("text/plain; charset=utf-8")
        .body(body.text.clone()))
}

#[post("/download/docx")]
async fn download_docx(body: web::Json<DownloadRequest>) -> actix_web::Result<HttpResponse> {
    let filename = sanitize_filename(body.filename.as_deref().unwrap_or("transcription"));
    let docx = build_docx(&body.text).map_err(error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok()
        .append_header((
            "Content-Disposition",
            format!("attachment; filename=\"{filename}.docx\""),
        ))
        .content_type("application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        .body(docx))
}

fn sanitize_filename(value: &str) -> String {
    let filtered: String = value
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .collect();

    if filtered.is_empty() {
        "transcription".to_string()
    } else {
        filtered
    }
}

fn build_docx(text: &str) -> anyhow::Result<Vec<u8>> {
    let cursor = std::io::Cursor::new(Vec::new());
    let mut zip = ZipWriter::new(cursor);
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);

    zip.start_file("[Content_Types].xml", options)?;
    zip.write_all(br#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"#)?;

    zip.add_directory("_rels/", options)?;
    zip.start_file("_rels/.rels", options)?;
    zip.write_all(br#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"#)?;

    zip.add_directory("word/", options)?;
    zip.start_file("word/document.xml", options)?;
    let escaped = xml_escape(text);
    let doc = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\
<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">\
<w:body><w:p><w:r><w:t xml:space=\"preserve\">{escaped}</w:t></w:r></w:p></w:body>\
</w:document>"
    );
    zip.write_all(doc.as_bytes())?;

    let bytes = zip.finish()?.into_inner();
    Ok(bytes)
}

fn xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .wrap(Cors::permissive())
            .service(health)
            .service(transcribe)
            .service(download_txt)
            .service(download_docx)
    })
    .bind(("0.0.0.0", 8080))?
    .run()
    .await
}

#[cfg(test)]
mod tests {
    use super::{build_docx, sanitize_filename, xml_escape};

    #[test]
    fn sanitizes_filename() {
        assert_eq!(sanitize_filename("audio output"), "audiooutput");
        assert_eq!(sanitize_filename("../bad"), "bad");
        assert_eq!(sanitize_filename(""), "transcription");
    }

    #[test]
    fn escapes_xml_content() {
        assert_eq!(xml_escape("a<b&c>"), "a&lt;b&amp;c&gt;");
    }

    #[test]
    fn docx_contains_document_xml() {
        let bytes = build_docx("hello world").expect("docx should build");
        let content = String::from_utf8_lossy(&bytes);
        assert!(content.contains("word/document.xml"));
    }
}
