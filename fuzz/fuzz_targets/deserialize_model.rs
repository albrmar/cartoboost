#![no_main]

use cartoboost_core::Model;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(text) = std::str::from_utf8(data) {
        let _ = serde_json::from_str::<Model>(text);
    }
});
