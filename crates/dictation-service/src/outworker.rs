//! Runs all keyboard and clipboard output on one thread, so the coordinator
//! never blocks on it and jobs keep their order.
use std::sync::mpsc::{self, Sender};
use std::thread::{self, JoinHandle};

use crate::activity;
use crate::output::{PasteOutcome, Paster};
use crate::typer::Typer;

pub enum Job {
    Paste(String),
    Type(String),
    Backspace(usize),
    /// Copies text to the clipboard without pasting.
    SetClipboard(String),
    /// Replied to once every earlier job has run.
    Flush(Sender<()>),
}

pub struct OutWorker {
    tx: Sender<Job>,
    handle: JoinHandle<()>,
}

impl OutWorker {
    pub fn spawn(
        mut paster: Box<dyn Paster>,
        mut typer: Box<dyn Typer>,
        notify: Box<dyn Fn(String) + Send>,
    ) -> Self {
        let (tx, rx) = mpsc::channel::<Job>();
        let handle = thread::Builder::new()
            .name("output".into())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    match job {
                        Job::Paste(text) => match paster.paste(&text) {
                            Ok(PasteOutcome::Pasted) => {}
                            Ok(PasteOutcome::ClipboardOnly(why)) => {
                                notify(format!("copied to clipboard, not pasted: {why}"))
                            }
                            Err(e) => notify(format!("paste failed: {e}")),
                        },
                        Job::Type(text) => {
                            activity::begin_typing();
                            if let Err(e) = typer.type_text(&text) {
                                notify(e);
                            }
                            activity::end_typing();
                        }
                        Job::Backspace(n) => {
                            activity::begin_typing();
                            if let Err(e) = typer.backspace(n) {
                                notify(e);
                            }
                            activity::end_typing();
                        }
                        Job::SetClipboard(text) => {
                            if let Err(e) = paster.set_clipboard(&text) {
                                notify(format!("clipboard failed: {e}"));
                            }
                        }
                        Job::Flush(done) => {
                            let _ = done.send(());
                        }
                    }
                }
            })
            .expect("spawn output thread");
        Self { tx, handle }
    }

    /// Queues a job and returns immediately.
    pub fn send(&self, job: Job) {
        let _ = self.tx.send(job);
    }

    /// Blocks until every earlier job has run.
    pub fn flush(&self) {
        let (tx, rx) = mpsc::channel();
        if self.tx.send(Job::Flush(tx)).is_ok() {
            let _ = rx.recv();
        }
    }

    /// Runs the remaining jobs, then stops the thread.
    pub fn shutdown(self) {
        drop(self.tx);
        let _ = self.handle.join();
    }
}
