#![warn(missing_docs, unused_imports)]

//! The Brakerski-Fan-Vercauteren homomorphic encryption scheme

mod ciphertext;
mod encoding;
mod keys;
mod ops;
mod parameters;
mod plaintext;
mod plaintext_vec;
mod proto;
mod rgsw_ciphertext;

pub mod traits;
pub use ciphertext::Ciphertext;
pub use encoding::Encoding;
pub use keys::{
	EvaluationKey, EvaluationKeyBuilder, GaloisKey, PublicKey, RelinearizationKey, SecretKey,
};
pub use ops::{dot_product_scalar, Multiplicator};
pub use parameters::{BfvParameters, BfvParametersBuilder};
pub use plaintext::Plaintext;
pub use plaintext_vec::PlaintextVec;
pub use proto::bfv::{
	EvaluationKey as EvaluationKeyProto, GaloisKey as GaloisKeyProto, PublicKey as PublicKeyProto,
	RelinearizationKey as RelinearizationKeyProto,
};
pub use rgsw_ciphertext::RGSWCiphertext;
