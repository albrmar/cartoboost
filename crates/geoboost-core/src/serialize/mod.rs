use crate::tree::Model;
use crate::Result;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

pub fn save_json(model: &Model, path: impl AsRef<Path>) -> Result<()> {
    let writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(writer, model)?;
    Ok(())
}

pub fn load_json(path: impl AsRef<Path>) -> Result<Model> {
    let reader = BufReader::new(File::open(path)?);
    Ok(serde_json::from_reader(reader)?)
}
