#![warn(missing_docs, unused_imports)]
//! Ring operations for moduli up to 62 bits.

pub mod ntt;
pub mod primes;

use crate::errors::{Error, Result};
use fhe_util::{is_prime, transcode_from_bytes, transcode_to_bytes};
use itertools::{izip, Itertools};
use num_bigint::BigUint;
use num_traits::cast::ToPrimitive;
use rand::{distributions::Uniform, CryptoRng, Rng, RngCore};
use std::{
	ops::{BitAnd, Mul, Shl, Shr},
	simd::{simd_swizzle, LaneCount, Simd, SimdOrd, SimdPartialOrd, SupportedLaneCount},
};

macro_rules! lane_unroll {
	($self:ident, $op:ident, $n:expr,  $a:expr, $($b:expr, $b0: ident, $bi: ident),*) => {
		macro_rules! tmp {
			($lane:literal) => {
				let (a0, _) = $a.as_chunks_mut();
				$(
					let ($b0, _) = $b.as_chunks();
				)*
				izip!(
					a0,
					$(
						$b0,
					)*
				).for_each(|(ai $(
					,$bi
				)*)| {
					*ai = $self
						.$op::<$lane>(Simd::from_array(*ai) $(
							,Simd::from_array(*$bi)
						)*)
						.to_array();
				});
			};
		}
		match $n {
			8 => {
				tmp!(8);
			}
			16 => {
				tmp!(16);
			}
			32 => {
				tmp!(32);
			}
			_ => {
				tmp!(64);
			}
		}
	};
}

/// Structure encapsulating an integer modulus up to 62 bits.
#[derive(Debug, Clone, PartialEq)]
pub struct Modulus {
	p: u64,
	nbits: usize,
	barrett_hi: u64,
	barrett_lo: u64,
	leading_zeros: u32,
	pub(crate) supports_opt: bool,
	distribution: Uniform<u64>,

	leading_zeros_u64: u64,
	bits: u64,
}

// We need to declare Eq manually because of the `Uniform` member.
impl Eq for Modulus {}

impl Modulus {
	/// Create a modulus from an integer of at most 62 bits.
	pub fn new(p: u64) -> Result<Self> {
		if p < 2 || (p >> 62) != 0 {
			Err(Error::InvalidModulus(p))
		} else {
			let barrett = ((BigUint::from(1u64) << 128usize) / p).to_u128().unwrap(); // 2^128 / p

			Ok(Self {
				p,
				nbits: 64 - p.leading_zeros() as usize,
				barrett_hi: (barrett >> 64) as u64,
				barrett_lo: barrett as u64,
				leading_zeros: p.leading_zeros(),
				supports_opt: primes::supports_opt(p),
				distribution: Uniform::from(0..p),
				leading_zeros_u64: p.leading_zeros() as u64,
				bits: 64 - p.leading_zeros() as u64,
			})
		}
	}

	/// Returns the value of the modulus.
	pub const fn modulus(&self) -> u64 {
		self.p
	}

	/// Performs the modular addition of a and b in constant time.
	/// Aborts if a >= p or b >= p in debug mode.
	pub const fn add(&self, a: u64, b: u64) -> u64 {
		debug_assert!(a < self.p && b < self.p);
		Self::reduce1(a + b, self.p)
	}

	/// Performs the modular addition of a and b in variable time.
	/// Aborts if a >= p or b >= p in debug mode.
	///
	/// # Safety
	/// This function is not constant time and its timing may reveal information
	/// about the values being added.
	pub const unsafe fn add_vt(&self, a: u64, b: u64) -> u64 {
		debug_assert!(a < self.p && b < self.p);
		Self::reduce1_vt(a + b, self.p)
	}

	/// Performs the modular subtraction of a and b in constant time.
	/// Aborts if a >= p or b >= p in debug mode.
	pub const fn sub(&self, a: u64, b: u64) -> u64 {
		debug_assert!(a < self.p && b < self.p);
		Self::reduce1(a + self.p - b, self.p)
	}

	/// Performs the modular subtraction of a and b in constant time.
	/// Aborts if a >= p or b >= p in debug mode.
	///
	/// # Safety
	/// This function is not constant time and its timing may reveal information
	/// about the values being subtracted.
	const unsafe fn sub_vt(&self, a: u64, b: u64) -> u64 {
		debug_assert!(a < self.p && b < self.p);
		Self::reduce1_vt(a + self.p - b, self.p)
	}

	/// Performs the modular multiplication of a and b in constant time.
	/// Aborts if a >= p or b >= p in debug mode.
	pub const fn mul(&self, a: u64, b: u64) -> u64 {
		debug_assert!(a < self.p && b < self.p);
		self.reduce_u128((a as u128) * (b as u128))
	}

	/// Performs the modular multiplication of a and b in constant time.
	/// Aborts if a >= p or b >= p in debug mode.
	///
	/// # Safety
	/// This function is not constant time and its timing may reveal information
	/// about the values being multiplied.
	const unsafe fn mul_vt(&self, a: u64, b: u64) -> u64 {
		debug_assert!(a < self.p && b < self.p);
		Self::reduce1_vt(self.lazy_reduce_u128((a as u128) * (b as u128)), self.p)
	}

	/// Optimized modular multiplication of a and b in constant time.
	///
	/// Aborts if a >= p or b >= p in debug mode.
	pub const fn mul_opt(&self, a: u64, b: u64) -> u64 {
		debug_assert!(self.supports_opt);
		debug_assert!(a < self.p && b < self.p);

		self.reduce_opt_u128((a as u128) * (b as u128))
	}

	/// Optimized modular multiplication of a and b in variable time.
	/// Aborts if a >= p or b >= p in debug mode.
	///
	/// # Safety
	/// This function is not constant time and its timing may reveal information
	/// about the values being multiplied.
	const unsafe fn mul_opt_vt(&self, a: u64, b: u64) -> u64 {
		debug_assert!(self.supports_opt);
		debug_assert!(a < self.p && b < self.p);

		self.reduce_opt_u128_vt((a as u128) * (b as u128))
	}

	/// Modular negation in constant time.
	///
	/// Aborts if a >= p in debug mode.
	pub const fn neg(&self, a: u64) -> u64 {
		debug_assert!(a < self.p);
		Self::reduce1(self.p - a, self.p)
	}

	/// Modular negation in variable time.
	/// Aborts if a >= p in debug mode.
	///
	/// # Safety
	/// This function is not constant time and its timing may reveal information
	/// about the value being negated.
	const unsafe fn neg_vt(&self, a: u64) -> u64 {
		debug_assert!(a < self.p);
		Self::reduce1_vt(self.p - a, self.p)
	}

	/// Compute the Shoup representation of a.
	///
	/// Aborts if a >= p in debug mode.
	pub const fn shoup(&self, a: u64) -> u64 {
		debug_assert!(a < self.p);

		(((a as u128) << 64) / (self.p as u128)) as u64
	}

	/// Shoup multiplication of a and b in constant time.
	///
	/// Aborts if b >= p or b_shoup != shoup(b) in debug mode.
	pub const fn mul_shoup(&self, a: u64, b: u64, b_shoup: u64) -> u64 {
		Self::reduce1(self.lazy_mul_shoup(a, b, b_shoup), self.p)
	}

	/// Shoup multiplication of a and b in variable time.
	/// Aborts if b >= p or b_shoup != shoup(b) in debug mode.
	///
	/// # Safety
	/// This function is not constant time and its timing may reveal information
	/// about the values being multiplied.
	const unsafe fn mul_shoup_vt(&self, a: u64, b: u64, b_shoup: u64) -> u64 {
		Self::reduce1_vt(self.lazy_mul_shoup(a, b, b_shoup), self.p)
	}

	/// Lazy Shoup multiplication of a and b in constant time.
	/// The output is in the interval [0, 2 * p).
	///
	/// Aborts if b >= p or b_shoup != shoup(b) in debug mode.
	pub const fn lazy_mul_shoup(&self, a: u64, b: u64, b_shoup: u64) -> u64 {
		debug_assert!(b < self.p);
		debug_assert!(b_shoup == self.shoup(b));

		let q = ((a as u128) * (b_shoup as u128)) >> 64;
		let r = ((a as u128) * (b as u128) - q * (self.p as u128)) as u64;

		debug_assert!(r < 2 * self.p);

		r
	}

	/// Modular addition of vectors in place in constant time.
	///
	/// Aborts if a and b differ in size, and if any of their values is >= p in
	/// debug mode.
	pub fn add_vec(&self, a: &mut [u64], b: &[u64]) {
		debug_assert_eq!(a.len(), b.len());

		izip!(a.iter_mut(), b.iter()).for_each(|(ai, bi)| *ai = self.add(*ai, *bi));
	}

	/// Modular addition of vectors in place in variable time.
	/// Aborts if a and b differ in size, and if any of their values is >= p in
	/// debug mode.
	///
	/// # Safety
	/// This function is not constant time and its timing may reveal information
	/// about the values being added.
	pub unsafe fn add_vec_vt(&self, a: &mut [u64], b: &[u64]) {
		let n = a.len();
		debug_assert_eq!(n, b.len());

		let p = self.p;
		macro_rules! add_at {
			($idx:expr) => {
				*a.get_unchecked_mut($idx) =
					Self::reduce1_vt(*a.get_unchecked_mut($idx) + *b.get_unchecked($idx), p);
			};
		}

		if n % 16 == 0 {
			for i in 0..n / 16 {
				add_at!(16 * i);
				add_at!(16 * i + 1);
				add_at!(16 * i + 2);
				add_at!(16 * i + 3);
				add_at!(16 * i + 4);
				add_at!(16 * i + 5);
				add_at!(16 * i + 6);
				add_at!(16 * i + 7);
				add_at!(16 * i + 8);
				add_at!(16 * i + 9);
				add_at!(16 * i + 10);
				add_at!(16 * i + 11);
				add_at!(16 * i + 12);
				add_at!(16 * i + 13);
				add_at!(16 * i + 14);
				add_at!(16 * i + 15);
			}
		} else {
			izip!(a.iter_mut(), b.iter()).for_each(|(ai, bi)| *ai = self.add_vt(*ai, *bi));
		}
	}

	/// Modular subtraction of vectors in place in constant time.
	///
	/// Aborts if a and b differ in size, and if any of their values is >= p in
	/// debug mode.
	pub fn sub_vec(&self, a: &mut [u64], b: &[u64]) {
		debug_assert_eq!(a.len(), b.len());

		izip!(a.iter_mut(), b.iter()).for_each(|(ai, bi)| *ai = self.sub(*ai, *bi));
	}

	/// Modular subtraction of vectors in place in variable time.
	/// Aborts if a and b differ in size, and if any of their values is >= p in
	/// debug mode.
	///
	/// # Safety
	/// This function is not constant time and its timing may reveal information
	/// about the values being subtracted.
	pub unsafe fn sub_vec_vt(&self, a: &mut [u64], b: &[u64]) {
		let n = a.len();
		debug_assert_eq!(n, b.len());

		let p = self.p;
		macro_rules! sub_at {
			($idx:expr) => {
				*a.get_unchecked_mut($idx) =
					Self::reduce1_vt(p + *a.get_unchecked_mut($idx) - *b.get_unchecked($idx), p);
			};
		}

		if n % 16 == 0 {
			for i in 0..n / 16 {
				sub_at!(16 * i);
				sub_at!(16 * i + 1);
				sub_at!(16 * i + 2);
				sub_at!(16 * i + 3);
				sub_at!(16 * i + 4);
				sub_at!(16 * i + 5);
				sub_at!(16 * i + 6);
				sub_at!(16 * i + 7);
				sub_at!(16 * i + 8);
				sub_at!(16 * i + 9);
				sub_at!(16 * i + 10);
				sub_at!(16 * i + 11);
				sub_at!(16 * i + 12);
				sub_at!(16 * i + 13);
				sub_at!(16 * i + 14);
				sub_at!(16 * i + 15);
			}
		} else {
			izip!(a.iter_mut(), b.iter()).for_each(|(ai, bi)| *ai = self.sub_vt(*ai, *bi));
		}
	}

	/// Modular multiplication of vectors in place in constant time.
	///
	/// Aborts if a and b differ in size, and if any of their values is >= p in
	/// debug mode.
	pub fn mul_vec(&self, a: &mut [u64], b: &[u64]) {
		debug_assert_eq!(a.len(), b.len());

		if self.supports_opt {
			izip!(a.iter_mut(), b.iter()).for_each(|(ai, bi)| *ai = self.mul_opt(*ai, *bi));
		} else {
			izip!(a.iter_mut(), b.iter()).for_each(|(ai, bi)| *ai = self.mul(*ai, *bi));
		}
	}

	/// Modular scalar multiplication of vectors in place in constant time.
	///
	/// Aborts if any of the values in a is >= p in debug mode.
	pub fn scalar_mul_vec(&self, a: &mut [u64], b: u64) {
		let b_shoup = self.shoup(b);
		a.iter_mut()
			.for_each(|ai| *ai = self.mul_shoup(*ai, b, b_shoup));
	}

	/// Modular scalar multiplication of vectors in place in variable time.
	/// Aborts if any of the values in a is >= p in debug mode.
	///
	/// # Safety
	/// This function is not constant time and its timing may reveal information
	/// about the values being multiplied.
	pub unsafe fn scalar_mul_vec_vt(&self, a: &mut [u64], b: u64) {
		let b_shoup = self.shoup(b);
		a.iter_mut()
			.for_each(|ai| *ai = self.mul_shoup_vt(*ai, b, b_shoup));
	}

	/// Modular multiplication of vectors in place in variable time.
	/// Aborts if a and b differ in size, and if any of their values is >= p in
	/// debug mode.
	///
	/// # Safety
	/// This function is not constant time and its timing may reveal information
	/// about the values being subtracted.
	pub unsafe fn mul_vec_vt(&self, a: &mut [u64], b: &[u64]) {
		debug_assert_eq!(a.len(), b.len());

		if self.supports_opt {
			izip!(a.iter_mut(), b.iter()).for_each(|(ai, bi)| *ai = self.mul_opt_vt(*ai, *bi));
		} else {
			izip!(a.iter_mut(), b.iter()).for_each(|(ai, bi)| *ai = self.mul_vt(*ai, *bi));
		}
	}

	/// Compute the Shoup representation of a vector.
	///
	/// Aborts if any of the values of the vector is >= p in debug mode.
	pub fn shoup_vec(&self, a: &[u64]) -> Vec<u64> {
		a.iter().map(|ai| self.shoup(*ai)).collect_vec()
	}

	/// Shoup modular multiplication of vectors in place in constant time.
	///
	/// Aborts if a and b differ in size, and if any of their values is >= p in
	/// debug mode.
	pub fn mul_shoup_vec(&self, a: &mut [u64], b: &[u64], b_shoup: &[u64]) {
		debug_assert_eq!(a.len(), b.len());
		debug_assert_eq!(a.len(), b_shoup.len());
		debug_assert_eq!(&b_shoup, &self.shoup_vec(b));

		izip!(a.iter_mut(), b.iter(), b_shoup.iter())
			.for_each(|(ai, bi, bi_shoup)| *ai = self.mul_shoup(*ai, *bi, *bi_shoup));
	}

	/// Shoup modular multiplication of vectors in place in variable time.
	/// Aborts if a and b differ in size, and if any of their values is >= p in
	/// debug mode.
	///
	/// # Safety
	/// This function is not constant time and its timing may reveal information
	/// about the values being multiplied.
	pub unsafe fn mul_shoup_vec_vt(&self, a: &mut [u64], b: &[u64], b_shoup: &[u64]) {
		debug_assert_eq!(a.len(), b.len());
		debug_assert_eq!(a.len(), b_shoup.len());
		debug_assert_eq!(&b_shoup, &self.shoup_vec(b));

		izip!(a.iter_mut(), b.iter(), b_shoup.iter())
			.for_each(|(ai, bi, bi_shoup)| *ai = self.mul_shoup_vt(*ai, *bi, *bi_shoup));
	}

	/// Reduce a vector in place in constant time.
	pub fn reduce_vec(&self, a: &mut [u64]) {
		a.iter_mut().for_each(|ai| *ai = self.reduce(*ai));
	}

	/// Center a value modulo p as i64 in variable time.
	/// TODO: To test and to make constant time?
	///
	/// # Safety
	/// This function is not constant time and its timing may reveal information
	/// about the value being centered.
	const unsafe fn center_vt(&self, a: u64) -> i64 {
		debug_assert!(a < self.p);

		if a >= self.p >> 1 {
			(a as i64) - (self.p as i64)
		} else {
			a as i64
		}
	}

	/// Center a vector in variable time.
	///
	/// # Safety
	/// This function is not constant time and its timing may reveal information
	/// about the values being centered.
	pub unsafe fn center_vec_vt(&self, a: &[u64]) -> Vec<i64> {
		a.iter().map(|ai| self.center_vt(*ai)).collect_vec()
	}

	/// Reduce a vector in place in variable time.
	///
	/// # Safety
	/// This function is not constant time and its timing may reveal information
	/// about the values being reduced.
	pub unsafe fn reduce_vec_vt(&self, a: &mut [u64]) {
		a.iter_mut().for_each(|ai| *ai = self.reduce_vt(*ai));
	}

	/// Modular reduction of a i64 in constant time.
	const fn reduce_i64(&self, a: i64) -> u64 {
		self.reduce_u128((((self.p as i128) << 64) + (a as i128)) as u128)
	}

	/// Modular reduction of a i64 in variable time.
	///
	/// # Safety
	/// This function is not constant time and its timing may reveal information
	/// about the values being reduced.
	const unsafe fn reduce_i64_vt(&self, a: i64) -> u64 {
		self.reduce_u128_vt((((self.p as i128) << 64) + (a as i128)) as u128)
	}

	/// Reduce a vector in place in constant time.
	pub fn reduce_vec_i64(&self, a: &[i64]) -> Vec<u64> {
		a.iter().map(|ai| self.reduce_i64(*ai)).collect_vec()
	}

	/// Reduce a vector in place in variable time.
	///
	/// # Safety
	/// This function is not constant time and its timing may reveal information
	/// about the values being reduced.
	pub unsafe fn reduce_vec_i64_vt(&self, a: &[i64]) -> Vec<u64> {
		a.iter().map(|ai| self.reduce_i64_vt(*ai)).collect_vec()
	}

	/// Reduce a vector in constant time.
	pub fn reduce_vec_new(&self, a: &[u64]) -> Vec<u64> {
		a.iter().map(|ai| self.reduce(*ai)).collect_vec()
	}

	/// Reduce a vector in variable time.
	///
	/// # Safety
	/// This function is not constant time and its timing may reveal information
	/// about the values being reduced.
	pub unsafe fn reduce_vec_new_vt(&self, a: &[u64]) -> Vec<u64> {
		a.iter().map(|bi| self.reduce_vt(*bi)).collect_vec()
	}

	/// Modular negation of a vector in place in constant time.
	///
	/// Aborts if any of the values in the vector is >= p in debug mode.
	pub fn neg_vec(&self, a: &mut [u64]) {
		izip!(a.iter_mut()).for_each(|ai| *ai = self.neg(*ai));
	}

	/// Modular negation of a vector in place in variable time.
	/// Aborts if any of the values in the vector is >= p in debug mode.
	///
	/// # Safety
	/// This function is not constant time and its timing may reveal information
	/// about the values being negated.
	pub unsafe fn neg_vec_vt(&self, a: &mut [u64]) {
		izip!(a.iter_mut()).for_each(|ai| *ai = self.neg_vt(*ai));
	}

	/// Modular exponentiation in variable time.
	///
	/// Aborts if a >= p or n >= p in debug mode.
	pub fn pow(&self, a: u64, n: u64) -> u64 {
		debug_assert!(a < self.p && n < self.p);

		if n == 0 {
			1
		} else if n == 1 {
			a
		} else {
			let mut r = a;
			let mut i = (62 - n.leading_zeros()) as isize;
			while i >= 0 {
				r = self.mul(r, r);
				if (n >> i) & 1 == 1 {
					r = self.mul(r, a);
				}
				i -= 1;
			}
			r
		}
	}

	/// Modular inversion in variable time.
	///
	/// Returns None if p is not prime or a = 0.
	/// Aborts if a >= p in debug mode.
	pub fn inv(&self, a: u64) -> std::option::Option<u64> {
		if !is_prime(self.p) || a == 0 {
			None
		} else {
			let r = self.pow(a, self.p - 2);
			debug_assert_eq!(self.mul(a, r), 1);
			Some(r)
		}
	}

	/// Modular reduction of a u128 in constant time.
	pub const fn reduce_u128(&self, a: u128) -> u64 {
		Self::reduce1(self.lazy_reduce_u128(a), self.p)
	}

	/// Modular reduction of a u128 in variable time.
	///
	/// # Safety
	/// This function is not constant time and its timing may reveal information
	/// about the value being reduced.
	pub const unsafe fn reduce_u128_vt(&self, a: u128) -> u64 {
		Self::reduce1_vt(self.lazy_reduce_u128(a), self.p)
	}

	/// Modular reduction of a u64 in constant time.
	pub const fn reduce(&self, a: u64) -> u64 {
		Self::reduce1(self.lazy_reduce(a), self.p)
	}

	/// Modular reduction of a u64 in variable time.
	///
	/// # Safety
	/// This function is not constant time and its timing may reveal information
	/// about the value being reduced.
	pub const unsafe fn reduce_vt(&self, a: u64) -> u64 {
		Self::reduce1_vt(self.lazy_reduce(a), self.p)
	}

	/// Optimized modular reduction of a u128 in constant time.
	pub const fn reduce_opt_u128(&self, a: u128) -> u64 {
		debug_assert!(self.supports_opt);
		Self::reduce1(self.lazy_reduce_opt_u128(a), self.p)
	}

	/// Optimized modular reduction of a u128 in constant time.
	///
	/// # Safety
	/// This function is not constant time and its timing may reveal information
	/// about the value being reduced.
	pub(crate) const unsafe fn reduce_opt_u128_vt(&self, a: u128) -> u64 {
		debug_assert!(self.supports_opt);
		Self::reduce1_vt(self.lazy_reduce_opt_u128(a), self.p)
	}

	/// Optimized modular reduction of a u64 in constant time.
	pub const fn reduce_opt(&self, a: u64) -> u64 {
		Self::reduce1(self.lazy_reduce_opt(a), self.p)
	}

	/// Optimized modular reduction of a u64 in variable time.
	///
	/// # Safety
	/// This function is not constant time and its timing may reveal information
	/// about the value being reduced.
	pub const unsafe fn reduce_opt_vt(&self, a: u64) -> u64 {
		Self::reduce1_vt(self.lazy_reduce_opt(a), self.p)
	}

	/// Return x mod p in constant time.
	/// Aborts if x >= 2 * p in debug mode.
	const fn reduce1(x: u64, p: u64) -> u64 {
		debug_assert!(p >> 63 == 0);
		debug_assert!(x < 2 * p);

		let (y, _) = x.overflowing_sub(p);
		let xp = x ^ p;
		let yp = y ^ p;
		let xy = xp ^ yp;
		let xxy = x ^ xy;
		let xxy = xxy >> 63;
		let (c, _) = xxy.overflowing_sub(1);
		let r = (c & y) | ((!c) & x);

		debug_assert!(r == x % p);

		r
	}

	/// Return x mod p in variable time.
	/// Aborts if x >= 2 * p in debug mode.
	///
	/// # Safety
	/// This function is not constant time and its timing may reveal information
	/// about the value being reduced.
	const unsafe fn reduce1_vt(x: u64, p: u64) -> u64 {
		debug_assert!(p >> 63 == 0);
		debug_assert!(x < 2 * p);

		if x >= p {
			x - p
		} else {
			x
		}
	}

	/// Lazy modular reduction of a in constant time.
	/// The output is in the interval [0, 2 * p).
	pub const fn lazy_reduce_u128(&self, a: u128) -> u64 {
		let a_lo = a as u64;
		let a_hi = (a >> 64) as u64;
		let p_lo_lo = ((a_lo as u128) * (self.barrett_lo as u128)) >> 64;
		let p_hi_lo = (a_hi as u128) * (self.barrett_lo as u128);
		let p_lo_hi = (a_lo as u128) * (self.barrett_hi as u128);

		let q = ((p_lo_hi + p_hi_lo + p_lo_lo) >> 64) + (a_hi as u128) * (self.barrett_hi as u128);
		let r = (a - q * (self.p as u128)) as u64;

		debug_assert!((r as u128) < 2 * (self.p as u128));
		debug_assert!(r % self.p == (a % (self.p as u128)) as u64);

		r
	}

	/// Lazy modular reduction of a in constant time.
	/// The output is in the interval [0, 2 * p).
	pub const fn lazy_reduce(&self, a: u64) -> u64 {
		let p_lo_lo = ((a as u128) * (self.barrett_lo as u128)) >> 64;
		let p_lo_hi = (a as u128) * (self.barrett_hi as u128);

		let q = (p_lo_hi + p_lo_lo) >> 64;
		let r = (a as u128 - q * (self.p as u128)) as u64;

		debug_assert!((r as u128) < 2 * (self.p as u128));
		debug_assert!(r % self.p == a % self.p);

		r
	}

	/// Lazy optimized modular reduction of a in constant time.
	/// The output is in the interval [0, 2 * p).
	///
	/// Aborts if the input is >= p ^ 2 in debug mode.
	pub const fn lazy_reduce_opt_u128(&self, a: u128) -> u64 {
		debug_assert!(a < (self.p as u128) * (self.p as u128));

		let q = (((self.barrett_lo as u128) * (a >> 64)) + (a << self.leading_zeros)) >> 64;
		let r = (a - q * (self.p as u128)) as u64;

		debug_assert!((r as u128) < 2 * (self.p as u128));
		debug_assert!(r % self.p == (a % (self.p as u128)) as u64);

		r
	}

	/// Lazy optimized modular reduction of a in constant time.
	/// The output is in the interval [0, 2 * p).
	const fn lazy_reduce_opt(&self, a: u64) -> u64 {
		let q = a >> (64 - self.leading_zeros);
		let r = ((a as u128) - (q as u128) * (self.p as u128)) as u64;

		debug_assert!((r as u128) < 2 * (self.p as u128));
		debug_assert!(r % self.p == a % self.p);

		r
	}

	/// Lazy modular reduction of a vector in constant time.
	/// The output coefficients are in the interval [0, 2 * p).
	pub fn lazy_reduce_vec(&self, a: &mut [u64]) {
		if self.supports_opt {
			a.iter_mut().for_each(|ai| *ai = self.lazy_reduce_opt(*ai))
		} else {
			a.iter_mut().for_each(|ai| *ai = self.lazy_reduce(*ai))
		}
	}

	/// Returns a random vector.
	pub fn random_vec<R: RngCore + CryptoRng>(&self, size: usize, rng: &mut R) -> Vec<u64> {
		rng.sample_iter(self.distribution).take(size).collect_vec()
	}

	/// Length of the serialization of a vector of size `size`.
	///
	/// Panics if the size is not a multiple of 8.
	pub const fn serialization_length(&self, size: usize) -> usize {
		assert!(size % 8 == 0);
		let p_nbits = 64 - (self.p - 1).leading_zeros() as usize;
		p_nbits * size / 8
	}

	/// Serialize a vector of elements of length a multiple of 8.
	///
	/// Panics if the length of the vector is not a multiple of 8.
	pub fn serialize_vec(&self, a: &[u64]) -> Vec<u8> {
		let p_nbits = 64 - (self.p - 1).leading_zeros() as usize;
		transcode_to_bytes(a, p_nbits)
	}

	/// Deserialize a vector of bytes into a vector of elements mod p.
	pub fn deserialize_vec(&self, b: &[u8]) -> Vec<u64> {
		let p_nbits = 64 - (self.p - 1).leading_zeros() as usize;
		transcode_from_bytes(b, p_nbits)
	}

	#[inline]
	pub fn lazy_reduce_simd<const LANES: usize>(&self, a: Simd<u64, LANES>) -> Simd<u64, LANES>
	where
		LaneCount<LANES>: SupportedLaneCount,
	{
		let barret_lo = Simd::splat(self.barrett_lo);
		let barret_hi = Simd::splat(self.barrett_hi);

		// (a * barret_lo) >> 64
		let p_lo_lo = self.mulhi_simd(a, barret_lo);

		// (a * barret_hi)
		let p_lo_hi_lo = a * barret_hi;
		let p_lo_hi_hi = self.mulhi_simd(a, barret_hi);

		// (p_lo_lo + p_lo_hi) >> 64
		let res = p_lo_hi_lo + p_lo_lo;
		let q = res
			.simd_lt(p_lo_hi_lo)
			.select(p_lo_hi_hi + Simd::splat(1), p_lo_hi_hi);

		a - q * Simd::splat(self.p)
	}

	#[inline]
	pub fn lazy_reduce_u128_simd<const LANES: usize>(
		&self,
		a_hi: Simd<u64, LANES>,
		a_lo: Simd<u64, LANES>,
	) -> Simd<u64, LANES>
	where
		LaneCount<LANES>: SupportedLaneCount,
	{
		let barret_lo = Simd::splat(self.barrett_lo);
		let barret_hi = Simd::splat(self.barrett_hi);

		// => q = ((a_lo + 2^64 a_hi) * (b_lo + 2^64 b_hi)) >> 128
		// => q = ((a_lo * b_lo) >> 64 + (a_lo * b_hi) + (b_lo * a_hi)) >> 64 + (a_hi *
		// b_hi) (a_lo * b_lo) >> 64

		let q_lo_lo = self.mulhi_simd(a_lo, barret_lo);
		// (a_lo * b_hi)_lo
		let q_lo_hi_lo = a_lo * barret_hi;
		// (a_hi * b_lo)_lo
		let q_hi_lo_lo = a_hi * barret_lo;

		let sum_lo = q_lo_lo + q_lo_hi_lo + q_hi_lo_lo;

		let sum_hi = self.mulhi_simd(a_hi, barret_lo) + self.mulhi_simd(a_lo, barret_hi);
		let sum_hi = sum_lo
			.simd_lt(q_hi_lo_lo)
			.select(sum_hi + Simd::splat(1), sum_hi);

		// (a_hi * b_hi)_lo
		let q_hi_hi_lo = a_hi * barret_hi;
		let q_lo = q_hi_hi_lo + sum_hi;

		a_lo - q_lo * Simd::splat(self.p)
	}

	#[inline]
	pub fn lazy_reduce_opt_simd<const LANES: usize>(&self, a: Simd<u64, LANES>) -> Simd<u64, LANES>
	where
		LaneCount<LANES>: SupportedLaneCount,
	{
		// a << 2^s0
		let q = a.shr(Simd::splat(self.bits));
		a - q * Simd::splat(self.p)
	}

	#[inline]
	pub fn lazy_reduce_opt_u128_simd<const LANES: usize>(
		&self,
		a_hi: Simd<u64, LANES>,
		a_lo: Simd<u64, LANES>,
	) -> Simd<u64, LANES>
	where
		LaneCount<LANES>: SupportedLaneCount,
	{
		// qt = barret_lo * a_hi
		let barret_lo = Simd::splat(self.barrett_lo);
		let qt_hi = self.mulhi_simd(barret_lo, a_hi);
		let qt_lo = barret_lo * a_hi;

		// qb = a << 2^s0
		let ls = Simd::splat(self.leading_zeros_u64);
		let qb_hi = a_hi.shl(ls) + a_lo.shr(Simd::splat(self.bits));
		let qb_lo = a_lo.shl(ls);

		// q = qt + qb
		let r_lo = qt_lo + qb_lo;
		let c_mask = r_lo.simd_lt(qt_lo);
		let q_hi = qt_hi + qb_hi;
		let q_hi = c_mask.select(q_hi + Simd::splat(1), q_hi);

		// r = a_lo - q_hi * p
		a_lo - q_hi * Simd::splat(self.p)
	}

	#[inline]
	pub fn reduce_u128_simd<const LANES: usize>(
		&self,
		a_hi: Simd<u64, LANES>,
		a_lo: Simd<u64, LANES>,
	) -> Simd<u64, LANES>
	where
		LaneCount<LANES>: SupportedLaneCount,
	{
		let r = self.lazy_reduce_u128_simd(a_hi, a_lo);
		r.simd_min(r - Simd::splat(self.p))
	}

	#[inline]
	pub fn reduce_opt_u128_simd<const LANES: usize>(
		&self,
		a_hi: Simd<u64, LANES>,
		a_lo: Simd<u64, LANES>,
	) -> Simd<u64, LANES>
	where
		LaneCount<LANES>: SupportedLaneCount,
	{
		debug_assert!(self.supports_opt);
		let r = self.lazy_reduce_opt_u128_simd(a_hi, a_lo);
		r.simd_min(r - Simd::splat(self.p))
	}

	#[inline]
	pub fn mulhi_simd<const LANES: usize>(
		&self,
		a: Simd<u64, LANES>,
		b: Simd<u64, LANES>,
	) -> Simd<u64, LANES>
	where
		LaneCount<LANES>: SupportedLaneCount,
	{
		let shift_32 = Simd::splat(32);
		let low_mask = Simd::splat(0xffffffffu64);

		let a_hi = a.shr(shift_32);
		let a_lo = a.bitand(low_mask);
		let b_hi = b.shr(shift_32);
		let b_lo = b.bitand(low_mask);

		// c = a * b
		let c_lo_lo = a_lo * b_lo;
		let c_hi_lo = a_hi * b_lo;
		let c_lo_hi = a_lo * b_hi;
		let c_hi_hi = a_hi * b_hi;

		// Calc c_hi

		// we don't need lower 32 bits of c_lo_lo for c_hi
		let c_lo_lo_shift = c_lo_lo.shr(shift_32);

		let s_mid = c_hi_lo + c_lo_lo_shift;
		let s_low = s_mid.bitand(low_mask);
		let s_mid = s_mid.shr(shift_32);
		let s_mid2 = (c_lo_hi + s_low).shr(shift_32);

		let mut c_hi = c_hi_hi + s_mid2;
		c_hi += s_mid;

		c_hi
	}

	#[inline]
	pub fn add_simd<const LANES: usize>(
		&self,
		a: Simd<u64, LANES>,
		b: Simd<u64, LANES>,
	) -> Simd<u64, LANES>
	where
		LaneCount<LANES>: SupportedLaneCount,
	{
		let c = a + b;
		c.simd_min(c - Simd::splat(self.p))
	}

	#[inline]
	pub fn sub_simd<const LANES: usize>(
		&self,
		a: Simd<u64, LANES>,
		b: Simd<u64, LANES>,
	) -> Simd<u64, LANES>
	where
		LaneCount<LANES>: SupportedLaneCount,
	{
		let c = a + self.neg_simd(b);
		c.simd_min(c - Simd::splat(self.p))
	}

	#[inline]
	pub fn neg_simd<const LANES: usize>(&self, a: Simd<u64, LANES>) -> Simd<u64, LANES>
	where
		LaneCount<LANES>: SupportedLaneCount,
	{
		let p = Simd::splat(self.p);
		let n = p - a;
		n.simd_min(n - p)
	}

	#[inline]
	pub fn mul_simd<const LANES: usize>(
		&self,
		a: Simd<u64, LANES>,
		b: Simd<u64, LANES>,
	) -> Simd<u64, LANES>
	where
		LaneCount<LANES>: SupportedLaneCount,
	{
		let r_hi = self.mulhi_simd(a, b);
		let r_lo = a * b;

		if self.supports_opt {
			self.reduce_opt_u128_simd(r_hi, r_lo)
		} else {
			self.reduce_u128_simd(r_hi, r_lo)
		}
	}

	#[inline]
	pub fn lazy_mul_shoup_simd<const LANES: usize>(
		&self,
		a: Simd<u64, LANES>,
		b: Simd<u64, LANES>,
		b_shoup: Simd<u64, LANES>,
	) -> Simd<u64, LANES>
	where
		LaneCount<LANES>: SupportedLaneCount,
	{
		let q = self.mulhi_simd(a, b_shoup);
		(a * b) - (q * Simd::splat(self.p))
	}

	#[inline]
	pub fn mul_shoup_simd<const LANES: usize>(
		&self,
		a: Simd<u64, LANES>,
		b: Simd<u64, LANES>,
		b_shoup: Simd<u64, LANES>,
	) -> Simd<u64, LANES>
	where
		LaneCount<LANES>: SupportedLaneCount,
	{
		let r = self.lazy_mul_shoup_simd(a, b, b_shoup);
		r.simd_min(r - Simd::splat(self.p))
	}

	pub fn add_vec_simd(&self, a: &mut [u64], b: &[u64], n: usize) {
		hexl_rs::elwise_add_mod(a, b, self.p, n as u64);
	}

	pub fn sub_vec_simd(&self, a: &mut [u64], b: &[u64], n: usize) {
		hexl_rs::elwise_sub_mod(a, b, self.p, n as u64);
	}

	pub fn neg_vec_simd(&self, a: &mut [u64], n: usize) {
		lane_unroll!(self, neg_simd, n, a,);
	}

	pub fn mul_vec_simd(&self, a: &mut [u64], b: &[u64], n: usize) {
		hexl_rs::elwise_mult_mod(a, b, self.p, n as u64, 1);
	}

	pub fn lazy_mul_shoup_vec_simd(&self, a: &mut [u64], b: &[u64], b_shoup: &[u64], n: usize) {
		if self.nbits > 52 {
			lane_unroll!(self, lazy_mul_shoup_simd, n, a, b, b0, bi, b_shoup, c0, ci);
		} else {
			hexl_rs::elwise_mult_mod(a, b, self.p, n as u64, 2);
		}
	}

	pub fn mul_shoup_vec_simd(&self, a: &mut [u64], b: &[u64], b_shoup: &[u64], n: usize) {
		if self.nbits > 52 {
			lane_unroll!(self, mul_shoup_simd, n, a, b, b0, bi, b_shoup, c0, ci);
		} else {
			hexl_rs::elwise_mult_mod(a, b, self.p, n as u64, 1);
		}
	}

	// pub fn reduce_opt_u128_vec_simd(&self, a: &mut [u64], b: &[u64], n: usize) {
	// 	lane_unroll!(self, reduce_opt_u128_simd, n, a, b, b0, bi);
	// }

	pub fn lazy_reduce_vec_simd(&self, a: &mut [u64], n: usize) {
		if self.nbits > 52 && self.supports_opt {
			lane_unroll!(self, lazy_reduce_opt_simd, n, a,);
		} else {
			hexl_rs::elem_reduce_mod(a, self.p, n as u64, self.p, 2);
		}
	}
}

#[cfg(test)]
mod tests {
	use std::simd::{Simd, SimdOrd};

	use crate::zq::primes::supports_opt;

	use super::primes::generate_prime;
	use super::{primes, Modulus};
	use fhe_util::catch_unwind;
	use itertools::{izip, Itertools};
	use proptest::collection::vec as prop_vec;
	use proptest::prelude::{any, BoxedStrategy, Just, Strategy};
	use rand::distributions::Uniform;
	use rand::prelude::Distribution;
	use rand::{thread_rng, RngCore};

	// Utility functions for the proptests.

	fn valid_moduli() -> impl Strategy<Value = Modulus> {
		any::<u64>().prop_filter_map("filter invalid moduli", |p| Modulus::new(p).ok())
	}

	fn vecs() -> BoxedStrategy<(Vec<u64>, Vec<u64>)> {
		prop_vec(any::<u64>(), 1..100)
			.prop_flat_map(|vec| {
				let len = vec.len();
				(Just(vec), prop_vec(any::<u64>(), len))
			})
			.boxed()
	}

	proptest! {
		#[test]
		fn constructor(p: u64) {
			// 63 and 64-bit integers do not work.
			prop_assert!(Modulus::new(p | (1u64 << 62)).is_err());
			prop_assert!(Modulus::new(p | (1u64 << 63)).is_err());

			// p = 0 & 1 do not work.
			prop_assert!(Modulus::new(0u64).is_err());
			prop_assert!(Modulus::new(1u64).is_err());

			// Otherwise, all moduli should work.
			prop_assume!(p >> 2 >= 2);
			let q = Modulus::new(p >> 2);
			prop_assert!(q.is_ok());
			prop_assert_eq!(q.unwrap().modulus(), p >> 2);
		}

		#[test]
		fn neg(p in valid_moduli(), mut a: u64,  mut q: u64) {
			a = p.reduce(a);
			prop_assert_eq!(p.neg(a), (p.modulus() - a) % p.modulus());
			unsafe { prop_assert_eq!(p.neg_vt(a), (p.modulus() - a) % p.modulus()) }

			q = (q % (u64::MAX - p.modulus())) + 1 + p.modulus(); // q > p
			prop_assert!(catch_unwind(|| p.neg(q)).is_err());
		}

		#[test]
		fn add(p in valid_moduli(), mut a: u64, mut b: u64, mut q: u64) {
			a = p.reduce(a);
			b = p.reduce(b);
			prop_assert_eq!(p.add(a, b), (a + b) % p.modulus());
			unsafe { prop_assert_eq!(p.add_vt(a, b), (a + b) % p.modulus()) }

			q = (q % (u64::MAX - p.modulus())) + 1 + p.modulus(); // q > p
			prop_assert!(catch_unwind(|| p.add(q, a)).is_err());
			prop_assert!(catch_unwind(|| p.add(a, q)).is_err());
		}

		#[test]
		fn sub(p in valid_moduli(), mut a: u64, mut b: u64, mut q: u64) {
			a = p.reduce(a);
			b = p.reduce(b);
			prop_assert_eq!(p.sub(a, b), (a + p.modulus() - b) % p.modulus());
			unsafe { prop_assert_eq!(p.sub_vt(a, b), (a + p.modulus() - b) % p.modulus()) }

			q = (q % (u64::MAX - p.modulus())) + 1 + p.modulus(); // q > p
			prop_assert!(catch_unwind(|| p.sub(q, a)).is_err());
			prop_assert!(catch_unwind(|| p.sub(a, q)).is_err());
		}

		#[test]
		fn mul(p in valid_moduli(), mut a: u64, mut b: u64, mut q: u64) {
			a = p.reduce(a);
			b = p.reduce(b);
			prop_assert_eq!(p.mul(a, b) as u128, ((a as u128) * (b as u128)) % (p.modulus() as u128));
			unsafe { prop_assert_eq!(p.mul_vt(a, b) as u128, ((a as u128) * (b as u128)) % (p.modulus() as u128)) }

			q = (q % (u64::MAX - p.modulus())) + 1 + p.modulus(); // q > p
			prop_assert!(catch_unwind(|| p.mul(q, a)).is_err());
			prop_assert!(catch_unwind(|| p.mul(a, q)).is_err());
		}

		#[test]
		fn mul_shoup(p in valid_moduli(), mut a: u64, mut b: u64, mut q: u64) {
			a = p.reduce(a);
			b = p.reduce(b);
			q = (q % (u64::MAX - p.modulus())) + 1 + p.modulus(); // q > p

			// Compute shoup representation
			let b_shoup = p.shoup(b);
			prop_assert!(catch_unwind(|| p.shoup(q)).is_err());

			// Check that the multiplication yields the expected result
			prop_assert_eq!(p.mul_shoup(a, b, b_shoup) as u128, ((a as u128) * (b as u128)) % (p.modulus() as u128));
			unsafe { prop_assert_eq!(p.mul_shoup_vt(a, b, b_shoup) as u128, ((a as u128) * (b as u128)) % (p.modulus() as u128)) }

			// Check that the multiplication with incorrect b_shoup panics in debug mode
			prop_assert!(catch_unwind(|| p.mul_shoup(a, q, b_shoup)).is_err());
			prop_assume!(a != b);
			prop_assert!(catch_unwind(|| p.mul_shoup(a, a, b_shoup)).is_err());
		}

		#[test]
		fn reduce(p in valid_moduli(), a: u64) {
			prop_assert_eq!(p.reduce(a), a % p.modulus());
			unsafe { prop_assert_eq!(p.reduce_vt(a), a % p.modulus()) }
			if p.supports_opt {
				prop_assert_eq!(p.reduce_opt(a), a % p.modulus());
				unsafe { prop_assert_eq!(p.reduce_opt_vt(a), a % p.modulus()) }
			}
		}

		#[test]
		fn lazy_reduce(p in valid_moduli(), a: u64) {
			prop_assert!(p.lazy_reduce(a) < 2 * p.modulus());
			prop_assert_eq!(p.lazy_reduce(a) % p.modulus(), p.reduce(a));
		}

		#[test]
		fn reduce_i64(p in valid_moduli(), a: i64) {
			let b = if a < 0 { p.neg(p.reduce(-a as u64)) } else { p.reduce(a as u64) };
			prop_assert_eq!(p.reduce_i64(a), b);
			unsafe { prop_assert_eq!(p.reduce_i64_vt(a), b) }
		}

		#[test]
		fn reduce_u128(p in valid_moduli(), mut a: u128) {
			prop_assert_eq!(p.reduce_u128(a) as u128, a % (p.modulus() as u128));
			unsafe { prop_assert_eq!(p.reduce_u128_vt(a) as u128, a % (p.modulus() as u128)) }
			if p.supports_opt {
				let p_square = (p.modulus() as u128) * (p.modulus() as u128);
				a %= p_square;
				prop_assert_eq!(p.reduce_opt_u128(a) as u128, a % (p.modulus() as u128));
				unsafe { prop_assert_eq!(p.reduce_opt_u128_vt(a) as u128, a % (p.modulus() as u128)) }
			}
		}

		#[test]
		fn add_vec(p in valid_moduli(), (mut a, mut b) in vecs()) {
			p.reduce_vec(&mut a);
			p.reduce_vec(&mut b);
			let c = a.clone();
			p.add_vec(&mut a, &b);
			prop_assert_eq!(a, izip!(b.iter(), c.iter()).map(|(bi, ci)| p.add(*bi, *ci)).collect_vec());
			a = c.clone();
			unsafe { p.add_vec_vt(&mut a, &b) }
			prop_assert_eq!(a, izip!(b.iter(), c.iter()).map(|(bi, ci)| p.add(*bi, *ci)).collect_vec());
		}

		#[test]
		fn sub_vec(p in valid_moduli(), (mut a, mut b) in vecs()) {
			p.reduce_vec(&mut a);
			p.reduce_vec(&mut b);
			let c = a.clone();
			p.sub_vec(&mut a, &b);
			prop_assert_eq!(a, izip!(b.iter(), c.iter()).map(|(bi, ci)| p.sub(*ci, *bi)).collect_vec());
			a = c.clone();
			unsafe { p.sub_vec_vt(&mut a, &b) }
			prop_assert_eq!(a, izip!(b.iter(), c.iter()).map(|(bi, ci)| p.sub(*ci, *bi)).collect_vec());
		}

		#[test]
		fn mul_vec(p in valid_moduli(), (mut a, mut b) in vecs()) {
			p.reduce_vec(&mut a);
			p.reduce_vec(&mut b);
			let c = a.clone();
			p.mul_vec(&mut a, &b);
			prop_assert_eq!(a, izip!(b.iter(), c.iter()).map(|(bi, ci)| p.mul(*ci, *bi)).collect_vec());
			a = c.clone();
			unsafe { p.mul_vec_vt(&mut a, &b); }
			prop_assert_eq!(a, izip!(b.iter(), c.iter()).map(|(bi, ci)| p.mul(*ci, *bi)).collect_vec());
		}

		#[test]
		fn scalar_mul_vec(p in valid_moduli(), mut a: Vec<u64>, mut b: u64) {
			p.reduce_vec(&mut a);
			b = p.reduce(b);
			let c = a.clone();

			p.scalar_mul_vec(&mut a, b);
			prop_assert_eq!(a, c.iter().map(|ci| p.mul(*ci, b)).collect_vec());

			a = c.clone();
			unsafe { p.scalar_mul_vec_vt(&mut a, b) }
			prop_assert_eq!(a, c.iter().map(|ci| p.mul(*ci, b)).collect_vec());
		}

		#[test]
		fn mul_shoup_vec(p in valid_moduli(), (mut a, mut b) in vecs()) {
			p.reduce_vec(&mut a);
			p.reduce_vec(&mut b);
			let b_shoup = p.shoup_vec(&b);
			let c = a.clone();
			p.mul_shoup_vec(&mut a, &b, &b_shoup);
			prop_assert_eq!(a, izip!(b.iter(), c.iter()).map(|(bi, ci)| p.mul(*ci, *bi)).collect_vec());
			a = c.clone();
			unsafe { p.mul_shoup_vec_vt(&mut a, &b, &b_shoup) }
			prop_assert_eq!(a, izip!(b.iter(), c.iter()).map(|(bi, ci)| p.mul(*ci, *bi)).collect_vec());
		}

		#[test]
		fn reduce_vec(p in valid_moduli(), a: Vec<u64>) {
			let mut b = a.clone();
			p.reduce_vec(&mut b);
			prop_assert_eq!(b, a.iter().map(|ai| p.reduce(*ai)).collect_vec());

			b = a.clone();
			unsafe { p.reduce_vec_vt(&mut b) }
			prop_assert_eq!(b, a.iter().map(|ai| p.reduce(*ai)).collect_vec());
		}

		#[test]
		fn lazy_reduce_vec(p in valid_moduli(), a: Vec<u64>) {
			let mut b = a.clone();
			p.lazy_reduce_vec(&mut b);
			prop_assert!(b.iter().all(|bi| *bi < 2 * p.modulus()));
			prop_assert!(izip!(a, b).all(|(ai, bi)| bi % p.modulus() == ai % p.modulus()));
		}

		#[test]
		fn reduce_vec_new(p in valid_moduli(), a: Vec<u64>) {
			let b = p.reduce_vec_new(&a);
			prop_assert_eq!(b, a.iter().map(|ai| p.reduce(*ai)).collect_vec());
			prop_assert_eq!(p.reduce_vec_new(&a), unsafe { p.reduce_vec_new_vt(&a) });
		}

		#[test]
		fn reduce_vec_i64(p in valid_moduli(), a: Vec<i64>) {
			let b = p.reduce_vec_i64(&a);
			prop_assert_eq!(b, a.iter().map(|ai| p.reduce_i64(*ai)).collect_vec());
			let b = unsafe { p.reduce_vec_i64_vt(&a) };
			prop_assert_eq!(b, a.iter().map(|ai| p.reduce_i64(*ai)).collect_vec());
		}

		#[test]
		fn neg_vec(p in valid_moduli(), mut a: Vec<u64>) {
			p.reduce_vec(&mut a);
			let mut b = a.clone();
			p.neg_vec(&mut b);
			prop_assert_eq!(b, a.iter().map(|ai| p.neg(*ai)).collect_vec());
			b = a.clone();
			unsafe { p.neg_vec_vt(&mut b); }
			prop_assert_eq!(b, a.iter().map(|ai| p.neg(*ai)).collect_vec());
		}

		#[test]
		fn random_vec(p in valid_moduli(), size in 1..1000usize) {
			let mut rng = thread_rng();

			let v = p.random_vec(size, &mut rng);
			prop_assert_eq!(v.len(), size);

			let w = p.random_vec(size, &mut rng);
			prop_assert_eq!(w.len(), size);

			if p.modulus().leading_zeros() <= 30 {
				prop_assert_ne!(v, w); // This will hold with probability at least 2^(-30)
			}
		}

		#[test]
		fn serialize(p in valid_moduli(), mut a in prop_vec(any::<u64>(), 8)) {
			p.reduce_vec(&mut a);
			let b = p.serialize_vec(&a);
			let c = p.deserialize_vec(&b);
			prop_assert_eq!(a, c);
		}
	}

	// TODO: Make a proptest.
	#[test]
	fn mul_opt() {
		let ntests = 100;
		let mut rng = rand::thread_rng();

		#[allow(clippy::single_element_loop)]
		for p in [4611686018326724609] {
			let q = Modulus::new(p).unwrap();
			assert!(primes::supports_opt(p));

			assert_eq!(q.mul_opt(0, 1), 0);
			assert_eq!(q.mul_opt(1, 1), 1);
			assert_eq!(q.mul_opt(2 % p, 3 % p), 6 % p);
			assert_eq!(q.mul_opt(p - 1, 1), p - 1);
			assert_eq!(q.mul_opt(p - 1, 2 % p), p - 2);

			assert!(catch_unwind(|| q.mul_opt(p, 1)).is_err());
			assert!(catch_unwind(|| q.mul_opt(p << 1, 1)).is_err());
			assert!(catch_unwind(|| q.mul_opt(0, p)).is_err());
			assert!(catch_unwind(|| q.mul_opt(0, p << 1)).is_err());

			for _ in 0..ntests {
				let a = rng.next_u64() % p;
				let b = rng.next_u64() % p;
				assert_eq!(
					q.mul_opt(a, b),
					(((a as u128) * (b as u128)) % (p as u128)) as u64
				);
			}
		}
	}

	// TODO: Make a proptest.
	#[test]
	fn pow() {
		let ntests = 10;
		let mut rng = rand::thread_rng();

		for p in [2u64, 3, 17, 1987, 4611686018326724609] {
			let q = Modulus::new(p).unwrap();

			assert_eq!(q.pow(p - 1, 0), 1);
			assert_eq!(q.pow(p - 1, 1), p - 1);
			assert_eq!(q.pow(p - 1, 2 % p), 1);
			assert_eq!(q.pow(1, p - 2), 1);
			assert_eq!(q.pow(1, p - 1), 1);

			assert!(catch_unwind(|| q.pow(p, 1)).is_err());
			assert!(catch_unwind(|| q.pow(p << 1, 1)).is_err());
			assert!(catch_unwind(|| q.pow(0, p)).is_err());
			assert!(catch_unwind(|| q.pow(0, p << 1)).is_err());

			for _ in 0..ntests {
				let a = rng.next_u64() % p;
				let b = (rng.next_u64() % p) % 1000;
				let mut c = b;
				let mut r = 1;
				while c > 0 {
					r = q.mul(r, a);
					c -= 1;
				}
				assert_eq!(q.pow(a, b), r);
			}
		}
	}

	// TODO: Make a proptest.
	#[test]
	fn inv() {
		let ntests = 100;
		let mut rng = rand::thread_rng();

		for p in [2u64, 3, 17, 1987, 4611686018326724609] {
			let q = Modulus::new(p).unwrap();

			assert!(q.inv(0).is_none());
			assert_eq!(q.inv(1).unwrap(), 1);
			assert_eq!(q.inv(p - 1).unwrap(), p - 1);

			assert!(catch_unwind(|| q.inv(p)).is_err());
			assert!(catch_unwind(|| q.inv(p << 1)).is_err());

			for _ in 0..ntests {
				let a = rng.next_u64() % p;
				let b = q.inv(a);

				if a == 0 {
					assert!(b.is_none())
				} else {
					assert!(b.is_some());
					assert_eq!(q.mul(a, b.unwrap()), 1)
				}
			}
		}
	}

	#[test]
	fn lazy_mul_shoup_simd_works() {
		let mut rng = thread_rng();
		let q = Modulus::new(4611686018326724609).unwrap();

		let vals = Uniform::new(0, q.p).sample_iter(rng).take(16).collect_vec();

		let (a, b) = vals.split_at(8);
		let b_shoup = q.shoup_vec(b.to_vec().as_slice());
		let product = izip!(a, b, b_shoup.iter())
			.map(|(_a, _b, _b_shoup)| q.mul_shoup(*_a, *_b, *_b_shoup))
			.collect_vec();

		let mut product_simd = q.lazy_mul_shoup_simd::<8>(
			Simd::from_slice(a),
			Simd::from_slice(b),
			Simd::from_slice(b_shoup.as_slice()),
		);
		// reduce [0, 2p) -> [0, p)
		product_simd = product_simd.simd_min(product_simd - Simd::splat(q.p));

		assert_eq!(product, product_simd.as_array());
	}

	#[test]
	fn lazy_reduce_simd_works() {
		let p = generate_prime(62, 1 << 8, 1 << 62).unwrap();
		let q_mod = Modulus::new(p).unwrap();

		let rng = thread_rng();
		let a = Uniform::new(1 << 61, u64::MAX)
			.sample_iter(rng)
			.take(8)
			.collect_vec();
		let lazy_reduce_simd = q_mod.lazy_reduce_simd::<8>(Simd::from_slice(a.as_slice()));
		let lazy_reduced = a.iter().map(|v| q_mod.lazy_reduce(*v)).collect_vec();

		assert_eq!(lazy_reduce_simd.to_array().to_vec(), lazy_reduced);
	}

	#[test]
	fn lazy_reduce_u128_opt_and_noopt_simd_works() {
		let q_mod = Modulus::new(4611686018326724609).unwrap();

		let rng = thread_rng();
		let a = Uniform::new(0u128, q_mod.p as u128 * q_mod.p as u128)
			.sample_iter(rng)
			.take(8)
			.collect_vec();

		let mut hi = a.iter().map(|v| (v >> 64) as u64).collect_vec();
		let mut lo = a.iter().map(|v| (v & ((1 << 64) - 1)) as u64).collect_vec();

		let lazy_reduced_simd = q_mod.lazy_reduce_u128_simd::<8>(
			Simd::from_slice(hi.as_slice()),
			Simd::from_slice(lo.as_slice()),
		);
		let lazy_reduced_opt_simd = q_mod.lazy_reduce_opt_u128_simd::<8>(
			Simd::from_slice(hi.as_slice()),
			Simd::from_slice(lo.as_slice()),
		);
		let lazy_reduced = a.iter().map(|v| q_mod.lazy_reduce_u128(*v)).collect_vec();

		// assert_eq!(lazy_reduced_opt_simd.to_array().to_vec(), lazy_reduced);
		assert_eq!(lazy_reduced_simd.to_array().to_vec(), lazy_reduced);
	}

	#[test]
	fn lazy_reduce_opt_and_noopt_simd_works() {
		let q_mod = Modulus::new(4611686018326724609).unwrap();

		let rng = thread_rng();
		let a = Uniform::new(1 << (64 - q_mod.p.leading_zeros()), u64::MAX)
			.sample_iter(rng)
			.take(8)
			.collect_vec();

		let lazy_reduced_simd = q_mod.lazy_reduce_simd::<8>(Simd::from_slice(a.as_slice()));
		let lazy_reduced_opt_simd = q_mod.lazy_reduce_opt_simd::<8>(Simd::from_slice(a.as_slice()));

		let lazy_reduced = a.iter().map(|v| q_mod.lazy_reduce(*v)).collect_vec();

		assert_eq!(lazy_reduced_opt_simd.to_array().to_vec(), lazy_reduced);
		assert_eq!(lazy_reduced_simd.to_array().to_vec(), lazy_reduced);
	}

	#[test]
	fn reduce_opt_128_simd_simulate() {
		let p = generate_prime(62, 1 << 8, 1 << 62).unwrap();
		let q_mod = Modulus::new(p).unwrap();

		let mut rng = thread_rng();
		let a = Uniform::new(0u128, p as u128 * p as u128).sample(&mut rng);

		let a_hi = (a >> 64) as u64;
		let a_lo = a as u64;

		let low_mask = 4294967295u64;

		// calc q
		let qt_hi = ((q_mod.barrett_lo as u128 * a_hi as u128) >> 64) as u64;
		let qt_lo = (q_mod.barrett_lo as u128 * a_hi as u128) as u64;

		// a_lo << 2^s0
		let qb_hi = (a_hi << q_mod.leading_zeros) + (a_lo >> (64 - q_mod.leading_zeros));
		let qb_lo = a_lo << (q_mod.leading_zeros);

		// calc c
		let tmp = (qb_lo) + (low_mask & qt_lo);
		let c = ((tmp >> 32) + (qt_lo >> 32)) >> 32;

		let tmp = qt_hi + c;
		let q = qb_hi + tmp;

		let res = (a_lo).wrapping_sub(q.wrapping_mul(p));

		assert_eq!(res, (a % p as u128) as u64);
	}
}
