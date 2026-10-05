use dictation_service::transcript::*;
use proptest::prelude::*;

fn apply(buf: &mut String, e: &Edit) -> usize {
    match e {
        Edit::Append(t) => {
            buf.push_str(t);
            0
        }
        Edit::Replace { backspaces, text } => {
            for _ in 0..*backspaces {
                buf.pop();
            }
            buf.push_str(text);
            *backspaces
        }
    }
}

fn pass_text(base: &[String], upto: usize, from: usize, mutate: usize, salt: usize) -> String {
    let mut w: Vec<String> = base[from..upto].to_vec();
    let n = w.len();
    for i in 0..mutate.min(n) {
        w[n - 1 - i] = format!("x{salt}y{i}");
    }
    w.join(" ")
}

proptest! {
    // Mutations only touch the last `lock - 1` words of a pass, so locked text
    // must never change and the final buffer must equal the true text.
    #[test]
    fn locked_text_is_never_touched(
        total in 3usize..25,
        lock in 2usize..5,
        steps in proptest::collection::vec((1usize..4, 0usize..4, 0usize..50), 1..30),
        append_after in proptest::option::of(0usize..30),
    ) {
        let base: Vec<String> = (0..total).map(|i| format!("מילה{i}")).collect();
        let mut t = Transcript::new(lock);
        let mut buf = String::new();
        let mut upto = 0;
        for (idx, (grow, mutate, salt)) in steps.into_iter().enumerate() {
            if append_after == Some(idx) {
                t.set_append_only();
            }
            upto = (upto + grow).min(total);
            let from = t.locked_words();
            let text = pass_text(&base, upto, from, mutate.min(lock - 1), salt);
            let locked_before = t.locked_text();
            let tail_before = buf.chars().count() - locked_before.chars().count();
            if let Some(e) = t.on_pass(&text) {
                let bs = apply(&mut buf, &e);
                prop_assert!(bs <= tail_before, "backspaces {bs} exceed tail {tail_before}");
            }
            prop_assert!(buf.starts_with(&locked_before), "locked text damaged");
            prop_assert!(buf.starts_with(&t.locked_text()), "new locked text not typed");
        }
        if append_after.is_none() {
            let from = t.locked_words();
            let (edit, full) = t.finish(&base[from..total].join(" "));
            if let Some(e) = edit {
                apply(&mut buf, &e);
            }
            prop_assert_eq!(&buf, &base.join(" "));
            prop_assert_eq!(&full, &base.join(" "));
        }
    }
}