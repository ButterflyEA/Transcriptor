use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use dictation_service::output::{PasteOutcome, Paster};
use dictation_service::outworker::{Job, OutWorker};
use dictation_service::typer::Typer;

type Log = Arc<Mutex<Vec<String>>>;

struct FakePaster(Log, Duration);
impl Paster for FakePaster {
    fn paste(&mut self, text: &str) -> Result<PasteOutcome, String> {
        std::thread::sleep(self.1);
        self.0.lock().unwrap().push(format!("paste:{text}"));
        Ok(PasteOutcome::Pasted)
    }
}

struct FakeTyper(Log);
impl Typer for FakeTyper {
    fn type_text(&mut self, s: &str) -> Result<(), String> {
        self.0.lock().unwrap().push(format!("type:{s}"));
        Ok(())
    }
    fn backspace(&mut self, n: usize) -> Result<(), String> {
        if n > 100 {
            return Err("too many".into());
        }
        self.0.lock().unwrap().push(format!("bs:{n}"));
        Ok(())
    }
    fn supports_backspace(&self) -> bool {
        true
    }
}

fn worker(delay: Duration) -> (OutWorker, Log, Arc<Mutex<Vec<String>>>) {
    let log: Log = Arc::default();
    let notes: Arc<Mutex<Vec<String>>> = Arc::default();
    let n = Arc::clone(&notes);
    let w = OutWorker::spawn(
        Box::new(FakePaster(log.clone(), delay)),
        Box::new(FakeTyper(log.clone())),
        Box::new(move |m| n.lock().unwrap().push(m)),
    );
    (w, log, notes)
}

#[test]
fn jobs_run_in_order_and_flush_waits() {
    let (w, log, _) = worker(Duration::from_millis(50));
    w.send(Job::Type("a".into()));
    w.send(Job::Backspace(1));
    w.send(Job::Paste("all".into()));
    w.send(Job::Type("b".into()));
    w.flush();
    assert_eq!(
        *log.lock().unwrap(),
        ["type:a", "bs:1", "paste:all", "type:b"]
    );
    w.shutdown();
}

#[test]
fn send_returns_immediately_even_when_output_is_slow() {
    let (w, log, _) = worker(Duration::from_millis(200));
    let t = Instant::now();
    w.send(Job::Paste("slow".into()));
    assert!(t.elapsed() < Duration::from_millis(100));
    w.flush();
    assert_eq!(log.lock().unwrap().len(), 1);
    w.shutdown();
}

#[test]
fn errors_are_reported_and_later_jobs_still_run() {
    let (w, log, notes) = worker(Duration::ZERO);
    w.send(Job::Backspace(1000));
    w.send(Job::Type("ok".into()));
    w.flush();
    assert_eq!(*notes.lock().unwrap(), ["too many"]);
    assert_eq!(*log.lock().unwrap(), ["type:ok"]);
    w.shutdown();
}

#[test]
fn shutdown_runs_queued_jobs_first() {
    let (w, log, _) = worker(Duration::from_millis(30));
    w.send(Job::Paste("last".into()));
    w.shutdown();
    assert_eq!(*log.lock().unwrap(), ["paste:last"]);
}
