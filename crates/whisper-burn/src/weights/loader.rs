use crate::weights::safetensors::{SafeTensors, slice_f32};
use crate::{Error, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use std::collections::HashSet;

pub struct WeightMap<B: Backend> {
    st: SafeTensors,
    bytes: Vec<u8>,
    device: B::Device,
    consumed: HashSet<String>,
}

impl<B: Backend> WeightMap<B> {
    pub fn new(st: SafeTensors, bytes: Vec<u8>, device: B::Device) -> Self {
        Self {
            st,
            bytes,
            device,
            consumed: HashSet::new(),
        }
    }

    pub fn names(&self) -> impl Iterator<Item = &str> + '_ {
        self.st.all_tensor_names()
    }

    pub fn is_consumed(&self, name: &str) -> bool {
        self.consumed.contains(name)
    }

    fn take_f32(&mut self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        if self.consumed.contains(name) {
            return Err(Error::MissingWeight(format!("{name} already consumed")));
        }
        let meta = self
            .st
            .by_name(name)
            .ok_or_else(|| Error::MissingWeight(name.to_string()))?;
        let vals = slice_f32(meta, &self.bytes)?.into_owned();
        Ok((vals, meta.shape.clone()))
    }

    pub fn take_1d(&mut self, name: &str, len: usize) -> Result<Tensor<B, 1>> {
        let (v, s) = self.take_f32(name)?;
        if s != vec![len] {
            return Err(Error::ShapeMismatch {
                name: name.into(),
                expected: vec![len],
                got: s,
            });
        }
        self.consumed.insert(name.to_string());
        Ok(Tensor::<B, 1>::from_data(
            TensorData::new(v, Shape::from([len])),
            &self.device,
        ))
    }

    pub fn take_2d(&mut self, name: &str, shape: [usize; 2]) -> Result<Tensor<B, 2>> {
        let (v, s) = self.take_f32(name)?;
        if s != shape.to_vec() {
            return Err(Error::ShapeMismatch {
                name: name.into(),
                expected: shape.to_vec(),
                got: s,
            });
        }
        self.consumed.insert(name.to_string());
        Ok(Tensor::<B, 2>::from_data(
            TensorData::new(v, Shape::from(shape)),
            &self.device,
        ))
    }

    pub fn take_3d(&mut self, name: &str, shape: [usize; 3]) -> Result<Tensor<B, 3>> {
        let (v, s) = self.take_f32(name)?;
        if s != shape.to_vec() {
            return Err(Error::ShapeMismatch {
                name: name.into(),
                expected: shape.to_vec(),
                got: s,
            });
        }
        self.consumed.insert(name.to_string());
        Ok(Tensor::<B, 3>::from_data(
            TensorData::new(v, Shape::from(shape)),
            &self.device,
        ))
    }

    /// Like `take_2d` but returns `Ok(None)` when the tensor is absent. Used for
    /// weights present in some checkpoint variants but not others.
    pub fn take_opt_2d(&mut self, name: &str, shape: [usize; 2]) -> Result<Option<Tensor<B, 2>>> {
        if self.st.by_name(name).is_none() {
            return Ok(None);
        }
        self.take_2d(name, shape).map(Some)
    }

    /// Like `take_1d` but returns `Ok(None)` when the tensor is absent. Used for
    /// weights present in some checkpoint variants but not others.
    pub fn take_opt_1d(&mut self, name: &str, len: usize) -> Result<Option<Tensor<B, 1>>> {
        if self.st.by_name(name).is_none() {
            return Ok(None);
        }
        self.take_1d(name, len).map(Some)
    }

    pub fn finish(&self) -> Result<()> {
        let remaining: Vec<String> = self
            .st
            .all_tensor_names()
            .filter(|n| !self.consumed.contains(*n))
            .map(|n| n.to_string())
            .collect();
        if remaining.is_empty() {
            Ok(())
        } else {
            Err(Error::UnexpectedWeight(remaining.join(", ")))
        }
    }
}
