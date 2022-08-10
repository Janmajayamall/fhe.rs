//! Plaintext type in the BFV encryption scheme.

use crate::parameters::BfvParameters;
use crate::traits::{Decoder, Encoder};
use math::rq::traits::TryConvertFrom;
use math::rq::{Poly, Representation};
use std::rc::Rc;
use zeroize::{Zeroize, ZeroizeOnDrop};

/// An encoding for the plaintext.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Encoding {
	/// A Poly encoding encodes a vector as coefficients of a polynomial;
	/// homomorphic operations are therefore polynomial operations.
	Poly,
	/// A Simd encoding encodes a vector so that homomorphic operations are
	/// component-wise operations on the coefficients of the underlying vectors. The
	/// Simd encoding require that the plaintext modulus is congruent to 1 modulo
	/// the degree of the underlying polynomial.
	Simd,
}

/// A plaintext object, that encodes a vector according to a specific encoding.
#[derive(Debug, Clone)]
pub struct Plaintext {
	/// The parameters of the underlying BFV encryption scheme.
	pub(crate) par: Rc<BfvParameters>,
	/// The value after encoding.
	pub(crate) value: Vec<i64>,
	/// The encoding of the plaintext, if known
	pub(crate) encoding: Option<Encoding>,
	/// The plaintext as a polynomial.
	pub(crate) poly_ntt: Poly,
}

// Equality.
impl PartialEq for Plaintext {
	fn eq(&self, other: &Self) -> bool {
		let mut eq = self.par == other.par && self.value == other.value;
		if self.encoding.is_some() && other.encoding.is_some() {
			eq &= self.encoding.as_ref().unwrap() == other.encoding.as_ref().unwrap()
		}
		eq
	}
}

impl Eq for Plaintext {}

// Conversions.
impl TryConvertFrom<&Plaintext> for Poly {
	type Error = String;

	fn try_convert_from<R>(
		pt: &Plaintext,
		ctx: &Rc<math::rq::Context>,
		_: R,
	) -> Result<Self, Self::Error>
	where
		R: Into<Option<Representation>>,
	{
		if ctx != &pt.par.ctx {
			Err("Incompatible contexts".to_string())
		} else {
			Poly::try_convert_from(&pt.value as &[i64], &pt.par.ctx, Representation::PowerBasis)
		}
	}
}

// Zeroizing of plaintexts.

impl Zeroize for Plaintext {
	fn zeroize(&mut self) {
		self.value.zeroize()
	}
}

impl ZeroizeOnDrop for Plaintext {}

// Encoding and decoding.

impl Encoder<&[u64]> for Plaintext {
	type Error = String;

	fn try_encode(
		value: &[u64],
		encoding: Encoding,
		par: &Rc<BfvParameters>,
	) -> Result<Self, Self::Error> {
		if value.len() > par.polynomial_degree {
			return Err("There are too many values to encode".to_string());
		}
		let mut v = vec![0u64; par.polynomial_degree];
		v[..value.len()].copy_from_slice(value);

		if encoding == Encoding::Simd {
			if let Some(op) = &par.op {
				op.backward(&mut v);
			} else {
				return Err("The plaintext does not allow for using the Simd encoding".to_string());
			}
		}

		let w = unsafe { par.plaintext.center_vec_vt(&v) };
		v.zeroize();

		let mut poly = Poly::try_convert_from(&w as &[i64], &par.ctx, Representation::PowerBasis)?;
		poly.change_representation(Representation::Ntt);

		Ok(Self {
			par: par.clone(),
			value: w,
			encoding: Some(encoding),
			poly_ntt: poly,
		})
	}
}

impl Encoder<&[i64]> for Plaintext {
	type Error = String;

	fn try_encode(
		value: &[i64],
		encoding: Encoding,
		par: &Rc<BfvParameters>,
	) -> Result<Self, Self::Error> {
		if value.len() > par.polynomial_degree {
			return Err("There are too many values to encode".to_string());
		}
		let mut v = vec![0u64; par.polynomial_degree];
		v[..value.len()].copy_from_slice(&par.plaintext.reduce_vec_i64(value));

		if encoding == Encoding::Simd {
			if let Some(op) = &par.op {
				op.backward(&mut v);
			} else {
				return Err("The plaintext does not allow for using the Simd encoding".to_string());
			}
		}

		let w = unsafe { par.plaintext.center_vec_vt(&v) };
		v.zeroize();

		let mut poly = Poly::try_convert_from(&w as &[i64], &par.ctx, Representation::PowerBasis)?;
		poly.change_representation(Representation::Ntt);

		Ok(Self {
			par: par.clone(),
			value: w,
			encoding: Some(encoding),
			poly_ntt: poly,
		})
	}
}

impl Decoder for Vec<u64> {
	type Error = String;

	fn try_decode<E>(pt: &Plaintext, encoding: E) -> Result<Vec<u64>, Self::Error>
	where
		E: Into<Option<Encoding>>,
	{
		let encoding = encoding.into();
		let enc: Encoding;
		if pt.encoding.is_none() && encoding.is_none() {
			return Err("No encoding specified".to_string());
		} else if pt.encoding.is_some() {
			enc = pt.encoding.as_ref().unwrap().clone();
			if let Some(arg_enc) = encoding && arg_enc != enc {
				return Err("Mismatched encodings".to_string())
			}
		} else {
			enc = encoding.unwrap();
			if let Some(pt_enc) = pt.encoding.as_ref() && pt_enc != &enc {
				return Err("Mismatched encodings".to_string())
			}
		}

		let mut w = pt.par.plaintext.reduce_vec_i64(&pt.value);
		if enc == Encoding::Simd {
			if let Some(op) = &pt.par.op {
				op.forward(&mut w);
				Ok(w)
			} else {
				Err("The plaintext does not allow for using the Simd encoding".to_string())
			}
		} else {
			Ok(w)
		}
	}
}

impl Decoder for Vec<i64> {
	type Error = String;

	fn try_decode<E>(pt: &Plaintext, encoding: E) -> Result<Vec<i64>, Self::Error>
	where
		E: Into<Option<Encoding>>,
	{
		let v = Vec::<u64>::try_decode(pt, encoding)?;
		Ok(unsafe { pt.par.plaintext.center_vec_vt(&v) })
	}
}

/// Encode a plaintext as a polynomial in NTT representation.
/// We use the encoding technique described in Sec 3.1 of https://eprint.iacr.org/2021/204.
pub(crate) fn encode_pt(
	par: &Rc<BfvParameters>,
	pt: &Plaintext,
	encoding: Option<Encoding>,
) -> Result<Poly, String> {
	let enc: Encoding;
	if let Some(encoding) = encoding {
		enc = encoding;
		if pt.encoding.is_some() && Some(enc.clone()) != pt.encoding {
			return Err("Mismatched encodings".to_string());
		}
	} else if pt.encoding.is_some() {
		enc = pt.encoding.clone().unwrap();
	} else {
		return Err("Missing encodings".to_string());
	}

	let mut m_v = Vec::<u64>::try_decode(pt, enc.clone())?;
	par.plaintext.scalar_mul_vec(&mut m_v, par.q_mod_t);
	let pt = Plaintext::try_encode(&m_v as &[u64], enc, par)?;
	m_v.zeroize();
	let mut m = Poly::try_convert_from(&pt, &par.ctx, Representation::PowerBasis)?;
	m.change_representation(Representation::Ntt);
	m *= &par.delta;
	Ok(m)
}

#[cfg(test)]
mod tests {
	use super::{Encoding, Plaintext};
	use crate::parameters::{BfvParameters, BfvParametersBuilder};
	use crate::traits::{Decoder, Encoder};
	use std::rc::Rc;

	#[test]
	fn try_encode() {
		// The default test parameters support both Poly and Simd encodings
		let params = Rc::new(BfvParameters::default(1));
		let a = params.plaintext.random_vec(params.polynomial_degree);

		let plaintext = Plaintext::try_encode(&[0u64; 9] as &[u64], Encoding::Poly, &params);
		assert!(plaintext.is_err());

		let plaintext = Plaintext::try_encode(&a as &[u64], Encoding::Poly, &params);
		assert!(plaintext.is_ok());

		let plaintext = Plaintext::try_encode(&a as &[u64], Encoding::Simd, &params);
		assert!(plaintext.is_ok());

		let plaintext = Plaintext::try_encode(&[1u64] as &[u64], Encoding::Poly, &params);
		assert!(plaintext.is_ok());

		// The following parameters do not allow for Simd encoding
		let params = Rc::new(
			BfvParametersBuilder::default()
				.polynomial_degree(8)
				.plaintext_modulus(2)
				.ciphertext_moduli(vec![4611686018326724609])
				.build()
				.unwrap(),
		);

		let a = params.plaintext.random_vec(params.polynomial_degree);

		let plaintext = Plaintext::try_encode(&a as &[u64], Encoding::Poly, &params);
		assert!(plaintext.is_ok());

		let plaintext = Plaintext::try_encode(&a as &[u64], Encoding::Simd, &params);
		assert!(plaintext.is_err());
	}

	#[test]
	fn test_encode_decode() {
		(0..100).for_each(|_| {
			let params = Rc::new(BfvParameters::default(1));
			let a = params.plaintext.random_vec(params.polynomial_degree);

			let plaintext = Plaintext::try_encode(&a as &[u64], Encoding::Poly, &params);
			assert!(plaintext.is_ok());
			let b = Vec::<u64>::try_decode(&plaintext.unwrap(), Encoding::Poly);
			assert!(b.is_ok_and(|b| b == &a));

			let plaintext = Plaintext::try_encode(&a as &[u64], Encoding::Simd, &params);
			assert!(plaintext.is_ok());
			let b = Vec::<u64>::try_decode(&plaintext.unwrap(), Encoding::Simd);
			assert!(b.is_ok_and(|b| b == &a));

			let a = unsafe { params.plaintext.center_vec_vt(&a) };
			let plaintext = Plaintext::try_encode(&a as &[i64], Encoding::Poly, &params);
			assert!(plaintext.is_ok());
			let b = Vec::<i64>::try_decode(&plaintext.unwrap(), Encoding::Poly);
			assert!(b.is_ok_and(|b| b == &a));

			let plaintext = Plaintext::try_encode(&a as &[i64], Encoding::Simd, &params);
			assert!(plaintext.is_ok());
			let b = Vec::<i64>::try_decode(&plaintext.unwrap(), Encoding::Simd);
			assert!(b.is_ok_and(|b| b == &a));
		})
	}

	#[test]
	fn test_partial_eq() -> Result<(), String> {
		let params = Rc::new(BfvParameters::default(1));
		let a = params.plaintext.random_vec(params.polynomial_degree);

		let plaintext = Plaintext::try_encode(&a as &[u64], Encoding::Poly, &params)?;
		let mut same_plaintext = Plaintext::try_encode(&a as &[u64], Encoding::Poly, &params)?;
		assert_eq!(plaintext, same_plaintext);

		// Equality also holds when there is no encoding specified. In this test, we use the fact that
		// we can set it to None directly, but such a partial plaintext will be created during
		// decryption since we do not specify the encoding at the time.
		same_plaintext.encoding = None;
		assert_eq!(plaintext, same_plaintext);

		Ok(())
	}
}