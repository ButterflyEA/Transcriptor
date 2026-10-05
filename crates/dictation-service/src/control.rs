use std::io::{self, BufRead, BufReader, Write};
use std::thread;

use interprocess::local_socket::{
    GenericFilePath, GenericNamespaced, ListenerOptions, Name, Stream, ToFsName, ToNsName,
    traits::{ListenerExt as _, Stream as _},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Command {
    Start,
    Stop,
    Toggle,
    Status,
    Reload,
}

impl Command {
    pub fn parse(line: &str) -> Option<Command> {
        match line.trim().to_ascii_lowercase().as_str() {
            "start" => Some(Command::Start),
            "stop" => Some(Command::Stop),
            "toggle" => Some(Command::Toggle),
            "status" => Some(Command::Status),
            "reload" => Some(Command::Reload),
            _ => None,
        }
    }

    pub fn as_line(&self) -> &'static str {
        match self {
            Command::Start => "start",
            Command::Stop => "stop",
            Command::Toggle => "toggle",
            Command::Status => "status",
            Command::Reload => "reload",
        }
    }
}

/// Windows: a named pipe called `dictation`. Linux: `$XDG_RUNTIME_DIR/dictation.sock`.
pub fn socket_name() -> String {
    #[cfg(windows)]
    {
        "dictation".to_string()
    }
    #[cfg(not(windows))]
    {
        let dir = std::env::var_os("XDG_RUNTIME_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        dir.join("dictation.sock").to_string_lossy().into_owned()
    }
}

fn to_name(name: &str) -> io::Result<Name<'static>> {
    let owned = name.to_owned();
    if cfg!(windows) {
        owned.to_ns_name::<GenericNamespaced>()
    } else {
        owned.to_fs_name::<GenericFilePath>()
    }
}

/// Blocking accept loop; run it on its own thread or as the main loop. Fails
/// with `AddrInUse` when another instance is already serving the socket.
pub fn serve(handler: impl Fn(Command) -> String + Send + Sync + 'static) -> io::Result<()> {
    serve_on(&socket_name(), handler)
}

pub fn serve_on(
    name: &str,
    handler: impl Fn(Command) -> String + Send + Sync + 'static,
) -> io::Result<()> {
    #[cfg(unix)]
    {
        if send_to(name, Command::Status).is_ok() {
            return Err(io::Error::new(
                io::ErrorKind::AddrInUse,
                "dictation is already running",
            ));
        }
        // No live server answered, so any leftover socket file is stale.
        let _ = std::fs::remove_file(name);
    }

    let listener = ListenerOptions::new()
        .name(to_name(name)?)
        .create_sync()
        .map_err(|e| {
            if e.kind() == io::ErrorKind::PermissionDenied || e.kind() == io::ErrorKind::AddrInUse {
                io::Error::new(io::ErrorKind::AddrInUse, "dictation is already running")
            } else {
                e
            }
        })?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(name, std::fs::Permissions::from_mode(0o600));
    }

    let handler = std::sync::Arc::new(handler);
    for conn in listener.incoming() {
        let Ok(conn) = conn else { continue };
        let handler = handler.clone();
        thread::spawn(move || {
            let _ = handle_connection(conn, &*handler);
        });
    }
    Ok(())
}

fn handle_connection(conn: Stream, handler: &dyn Fn(Command) -> String) -> io::Result<()> {
    let mut reader = BufReader::new(conn);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let reply = match Command::parse(&line) {
        Some(cmd) => handler(cmd),
        None => "error: unknown command".to_string(),
    };
    let conn = reader.get_mut();
    conn.write_all(reply.as_bytes())?;
    conn.write_all(b"\n")?;
    conn.flush()
}

pub fn send(cmd: Command) -> io::Result<String> {
    send_to(&socket_name(), cmd)
}

pub fn send_to(name: &str, cmd: Command) -> io::Result<String> {
    let conn = Stream::connect(to_name(name)?)?;
    let mut reader = BufReader::new(conn);
    {
        let w = reader.get_mut();
        w.write_all(cmd.as_line().as_bytes())?;
        w.write_all(b"\n")?;
        w.flush()?;
    }
    let mut reply = String::new();
    reader.read_line(&mut reply)?;
    Ok(reply.trim().to_string())
}
