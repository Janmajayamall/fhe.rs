//! Number-Theoretic Transform in ZZ_q.

use std::{
	ops::BitAnd,
	simd::{
		simd_swizzle, u64x8, LaneCount, Simd, SimdOrd, SimdPartialEq, SimdPartialOrd,
		SupportedLaneCount,
		Which::{First, Second},
	},
	vec,
};

use super::Modulus;
use fhe_util::is_prime;
use hexl_rs::Ntt;
use itertools::izip;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Returns whether a modulus p is prime and supports the Number Theoretic
/// Transform of size n.
///
/// Aborts if n is not a power of 2 that is >= 8.
pub fn supports_ntt(p: u64, n: usize) -> bool {
	assert!(n >= 8 && n.is_power_of_two());

	p % ((n as u64) << 1) == 1 && is_prime(p)
}

/// Number-Theoretic Transform operator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NttOperator {
	p: Modulus,
	p_twice: u64,
	size: usize,
	omegas: Box<[u64]>,
	omegas_shoup: Box<[u64]>,
	omegas_inv: Box<[u64]>,
	zetas_inv: Box<[u64]>,
	zetas_inv_shoup: Box<[u64]>,
	size_inv: u64,
	size_inv_shoup: u64,
	ntt_hexl: hexl_rs::Ntt,
}

macro_rules! lane_unroll {
	($self:ident, $op:ident, $roots:ident, $roots_shoup:ident, $l:expr, $m:expr, $k:expr, $a_ptr:expr) => {
		macro_rules! tmp {
			($lane:literal) => {
				for i in 0..$m {
					let omega = Simd::splat(*$self.$roots.get_unchecked($k));
					let omega_shoup = Simd::splat(*$self.$roots_shoup.get_unchecked($k));
					$k += 1;

					let s = 2 * i * $l;

					let (x, _) = std::slice::from_raw_parts_mut($a_ptr.add(s), $l).as_chunks_mut();
					let (y, _) =
						std::slice::from_raw_parts_mut($a_ptr.add(s + $l), $l).as_chunks_mut();
					izip!(x, y).for_each(|(_x, _y)| {
						let (xr, yr) = $self.$op::<$lane>(
							Simd::from_array(*_x),
							Simd::from_array(*_y),
							omega,
							omega_shoup,
						);
						*_x = *xr.as_array();
						*_y = *yr.as_array();
					});
				}
			};
		}

		match $l {
			1 => {
				tmp!(1)
			}
			2 => {
				tmp!(2)
			}
			4 => {
				tmp!(4)
			}
			8 => {
				tmp!(8)
			}
			16 => {
				tmp!(16)
			}
			32 => {
				tmp!(32)
			}
			_ => tmp!(64),
		}
	};
}

impl NttOperator {
	/// Create an NTT operator given a modulus for a specific size.
	///
	/// Aborts if the size is not a power of 2 that is >= 8 in debug mode.
	/// Returns None if the modulus does not support the NTT for this specific
	/// size.
	pub fn new(p: &Modulus, size: usize) -> Option<Self> {
		if !supports_ntt(p.p, size) {
			None
		} else {
			let omega = Self::primitive_root(size, p);
			let omega_inv = p.inv(omega).unwrap();

			let mut exp = 1u64;
			let mut exp_inv = 1u64;
			let mut powers = Vec::with_capacity(size + 1);
			let mut powers_inv = Vec::with_capacity(size + 1);
			for _ in 0..size + 1 {
				powers.push(exp);
				powers_inv.push(exp_inv);
				exp = p.mul(exp, omega);
				exp_inv = p.mul(exp_inv, omega_inv);
			}

			let mut omegas = Vec::with_capacity(size);
			let mut omegas_inv = Vec::with_capacity(size);
			let mut zetas_inv = Vec::with_capacity(size);
			for i in 0..size {
				let j = i.reverse_bits() >> (size.leading_zeros() + 1);
				omegas.push(powers[j]);
				omegas_inv.push(powers_inv[j]);
				zetas_inv.push(powers_inv[j + 1]);
			}

			let size_inv = p.inv(size as u64).unwrap();

			let omegas_shoup = p.shoup_vec(&omegas);
			let zetas_inv_shoup = p.shoup_vec(&zetas_inv);

			let ntt_hexl = hexl_rs::Ntt::new(size as u64, p.p);

			Some(Self {
				p: p.clone(),
				p_twice: p.p * 2,
				size,
				omegas: omegas.into_boxed_slice(),
				omegas_shoup: omegas_shoup.into_boxed_slice(),
				omegas_inv: omegas_inv.into_boxed_slice(),
				zetas_inv: zetas_inv.into_boxed_slice(),
				zetas_inv_shoup: zetas_inv_shoup.into_boxed_slice(),
				size_inv,
				size_inv_shoup: p.shoup(size_inv),
				ntt_hexl,
			})
		}
	}

	/// Compute the forward NTT in place.
	/// Aborts if a is not of the size handled by the operator.
	pub fn forward(&self, a: &mut [u64]) {
		debug_assert_eq!(a.len(), self.size);

		let n = self.size;
		let a_ptr = a.as_mut_ptr();

		let mut l = n >> 1;
		let mut m = 1;
		let mut k = 1;
		while l > 0 {
			for i in 0..m {
				unsafe {
					let omega = *self.omegas.get_unchecked(k);
					let omega_shoup = *self.omegas_shoup.get_unchecked(k);
					k += 1;

					let s = 2 * i * l;
					match l {
						1 => {
							// The last level should reduce the output
							let uj = &mut *a_ptr.add(s);
							let ujl = &mut *a_ptr.add(s + l);
							self.butterfly(uj, ujl, omega, omega_shoup);
							*uj = self.reduce3(*uj);
							*ujl = self.reduce3(*ujl);
						}
						_ => {
							for j in s..(s + l) {
								self.butterfly(
									&mut *a_ptr.add(j),
									&mut *a_ptr.add(j + l),
									omega,
									omega_shoup,
								);
							}
						}
					}
				}
			}
			l >>= 1;
			m <<= 1;
		}
	}

	/// Compute the backward NTT in place.
	/// Aborts if a is not of the size handled by the operator.
	pub fn backward(&self, a: &mut [u64]) {
		debug_assert_eq!(a.len(), self.size);

		let a_ptr = a.as_mut_ptr();

		let mut k = 0;
		let mut m = self.size >> 1;
		let mut l = 1;
		while m > 0 {
			for i in 0..m {
				let s = 2 * i * l;
				unsafe {
					let zeta_inv = *self.zetas_inv.get_unchecked(k);
					let zeta_inv_shoup = *self.zetas_inv_shoup.get_unchecked(k);
					k += 1;
					match l {
						1 => {
							self.inv_butterfly(
								&mut *a_ptr.add(s),
								&mut *a_ptr.add(s + l),
								zeta_inv,
								zeta_inv_shoup,
							);
						}
						_ => {
							for j in s..(s + l) {
								self.inv_butterfly(
									&mut *a_ptr.add(j),
									&mut *a_ptr.add(j + l),
									zeta_inv,
									zeta_inv_shoup,
								);
							}
						}
					}
				}
			}
			l <<= 1;
			m >>= 1;
		}

		a.iter_mut()
			.for_each(|ai| *ai = self.p.mul_shoup(*ai, self.size_inv, self.size_inv_shoup));
	}

	/// Compute the forward NTT in place in variable time in a lazily fashion.
	/// This means that the output coefficients may be up to 4 times the
	/// modulus.
	///
	/// # Safety
	/// This function assumes that a_ptr points to at least `size` elements.
	/// This function is not constant time and its timing may reveal information
	/// about the value being reduced.
	pub unsafe fn forward_vt_lazy(&self, a_ptr: *mut u64) {
		let mut l = self.size >> 1;
		let mut m = 1;
		let mut k = 1;
		while l > 0 {
			for i in 0..m {
				let omega = *self.omegas.get_unchecked(k);
				let omega_shoup = *self.omegas_shoup.get_unchecked(k);
				k += 1;

				let s = 2 * i * l;
				match l {
					1 => {
						self.butterfly_vt(
							&mut *a_ptr.add(s),
							&mut *a_ptr.add(s + l),
							omega,
							omega_shoup,
						);
					}
					_ => {
						for j in s..(s + l) {
							self.butterfly_vt(
								&mut *a_ptr.add(j),
								&mut *a_ptr.add(j + l),
								omega,
								omega_shoup,
							);
						}
					}
				}
			}
			l >>= 1;
			m <<= 1;
		}
	}

	/// Compute the forward NTT in place in variable time.
	///
	/// # Safety
	/// This function assumes that a_ptr points to at least `size` elements.
	/// This function is not constant time and its timing may reveal information
	/// about the value being reduced.
	pub unsafe fn forward_vt(&self, a_ptr: *mut u64) {
		self.forward_vt_lazy(a_ptr);
		for i in 0..self.size {
			*a_ptr.add(i) = self.reduce3_vt(*a_ptr.add(i))
		}
	}

	/// Compute the backward NTT in place in variable time.
	///
	/// # Safety
	/// This function assumes that a_ptr points to at least `size` elements.
	/// This function is not constant time and its timing may reveal information
	/// about the value being reduced.
	pub unsafe fn backward_vt(&self, a_ptr: *mut u64) {
		let mut k = 0;
		let mut m = self.size >> 1;
		let mut l = 1;
		while m > 0 {
			for i in 0..m {
				let s = 2 * i * l;
				let zeta_inv = *self.zetas_inv.get_unchecked(k);
				let zeta_inv_shoup = *self.zetas_inv_shoup.get_unchecked(k);
				k += 1;
				match l {
					1 => {
						self.inv_butterfly_vt(
							&mut *a_ptr.add(s),
							&mut *a_ptr.add(s + l),
							zeta_inv,
							zeta_inv_shoup,
						);
					}
					_ => {
						for j in s..(s + l) {
							self.inv_butterfly_vt(
								&mut *a_ptr.add(j),
								&mut *a_ptr.add(j + l),
								zeta_inv,
								zeta_inv_shoup,
							);
						}
					}
				}
			}
			l <<= 1;
			m >>= 1;
		}

		for i in 0..self.size as isize {
			*a_ptr.offset(i) =
				self.p
					.mul_shoup(*a_ptr.offset(i), self.size_inv, self.size_inv_shoup)
		}
	}

	/// Reduce a modulo p.
	///
	/// Aborts if a >= 4 * p.
	const fn reduce3(&self, a: u64) -> u64 {
		debug_assert!(a < 4 * self.p.p);

		let y = Modulus::reduce1(a, 2 * self.p.p);
		Modulus::reduce1(y, self.p.p)
	}

	/// Reduce a modulo p in variable time.
	///
	/// Aborts if a >= 4 * p.
	const unsafe fn reduce3_vt(&self, a: u64) -> u64 {
		debug_assert!(a < 4 * self.p.p);

		let y = Modulus::reduce1_vt(a, 2 * self.p.p);
		Modulus::reduce1_vt(y, self.p.p)
	}

	/// NTT Butterfly.
	fn butterfly(&self, x: &mut u64, y: &mut u64, w: u64, w_shoup: u64) {
		debug_assert!(*x < 4 * self.p.p);
		debug_assert!(*y < 4 * self.p.p);
		debug_assert!(w < self.p.p);
		debug_assert_eq!(self.p.shoup(w), w_shoup);

		*x = Modulus::reduce1(*x, self.p_twice);
		let t = self.p.lazy_mul_shoup(*y, w, w_shoup);
		*y = *x + self.p_twice - t;
		*x += t;

		debug_assert!(*x < 4 * self.p.p);
		debug_assert!(*y < 4 * self.p.p);
	}

	/// NTT Butterfly in variable time.
	unsafe fn butterfly_vt(&self, x: &mut u64, y: &mut u64, w: u64, w_shoup: u64) {
		debug_assert!(*x < 4 * self.p.p);
		debug_assert!(*y < 4 * self.p.p);
		debug_assert!(w < self.p.p);
		debug_assert_eq!(self.p.shoup(w), w_shoup);

		*x = Modulus::reduce1_vt(*x, self.p_twice);
		let t = self.p.lazy_mul_shoup(*y, w, w_shoup);
		*y = *x + self.p_twice - t;
		*x += t;

		debug_assert!(*x < 4 * self.p.p);
		debug_assert!(*y < 4 * self.p.p);
	}

	/// Inverse NTT butterfly.
	fn inv_butterfly(&self, x: &mut u64, y: &mut u64, z: u64, z_shoup: u64) {
		debug_assert!(*x < self.p_twice);
		debug_assert!(*y < self.p_twice);
		debug_assert!(z < self.p.p);
		debug_assert_eq!(self.p.shoup(z), z_shoup);

		let t = *x;
		*x = Modulus::reduce1(*y + t, self.p_twice);
		*y = self.p.lazy_mul_shoup(self.p_twice + t - *y, z, z_shoup);

		debug_assert!(*x < self.p_twice);
		debug_assert!(*y < self.p_twice);
	}

	/// Inverse NTT butterfly in variable time
	unsafe fn inv_butterfly_vt(&self, x: &mut u64, y: &mut u64, z: u64, z_shoup: u64) {
		debug_assert!(*x < self.p_twice);
		debug_assert!(*y < self.p_twice);
		debug_assert!(z < self.p.p);
		debug_assert_eq!(self.p.shoup(z), z_shoup);

		let t = *x;
		*x = Modulus::reduce1_vt(*y + t, self.p_twice);
		*y = self.p.lazy_mul_shoup(self.p_twice + t - *y, z, z_shoup);

		debug_assert!(*x < self.p_twice);
		debug_assert!(*y < self.p_twice);
	}

	/// Returns a 2n-th primitive root modulo p.
	///
	/// Aborts if p is not prime or n is not a power of 2 that is >= 8.
	fn primitive_root(n: usize, p: &Modulus) -> u64 {
		debug_assert!(supports_ntt(p.p, n));

		let lambda = (p.p - 1) / (2 * n as u64);

		let mut rng: ChaCha8Rng = SeedableRng::seed_from_u64(0);
		for _ in 0..100 {
			let mut root = rng.gen_range(0..p.p);
			root = p.pow(root, lambda);
			if Self::is_primitive_root(root, 2 * n, p) {
				return root;
			}
		}

		debug_assert!(false, "Couldn't find primitive root");
		0
	}

	/// Returns whether a is a n-th primitive root of unity.
	///
	/// Aborts if a >= p in debug mode.
	fn is_primitive_root(a: u64, n: usize, p: &Modulus) -> bool {
		debug_assert!(a < p.p);
		debug_assert!(supports_ntt(p.p, n >> 1)); // TODO: This is not exactly the right condition here.

		// A primitive root of unity is such that x^n = 1 mod p, and x^(n/p) != 1 mod p
		// for all prime p dividing n.
		(p.pow(a, n as u64) == 1) && (p.pow(a, (n / 2) as u64) != 1)
	}

	pub fn forward_simd(&self, a: &mut [u64]) {
		// debug_assert!(LANES == 8);

		let n = self.size;
		let a_ptr = a.as_mut_ptr();

		let mut l = n >> 1;
		let mut m = 1;
		let mut k = 1;

		while l > 0 {
			unsafe {
				lane_unroll!(self, butterfly_simd, omegas, omegas_shoup, l, m, k, a_ptr);
			}
			l >>= 1;
			m <<= 1;
		}

		// reduce x in [0, 4p) to [0, p)
		let (x, _) = a.as_chunks_mut();
		let p_twice = Simd::splat(self.p_twice);
		let p = Simd::splat(self.p.p);
		x.iter_mut().for_each(|v: &mut [u64; 8]| {
			let mut _x = Simd::from_slice(v);
			_x = _x.simd_min(_x - p_twice);
			_x = _x.simd_min(_x - p);
			*v = *_x.as_array();
		});
	}

	pub fn forward_simd_swizzle(&self, a: &mut [u64]) {
		// debug_assert!(LANES == 8);

		let n = self.size;
		let a_ptr = a.as_mut_ptr();

		let mut l = n >> 1;
		let mut m = 1;
		let mut k = 1;

		while l > 0 {
			unsafe {
				match l {
					1 | 2 => {
						for i in 0..m {
							let omega = *self.omegas.get_unchecked(k);
							let omega_shoup = *self.omegas_shoup.get_unchecked(k);
							k += 1;

							let s = 2 * i * l;

							for j in s..s + l {
								self.butterfly(
									&mut *a_ptr.add(j),
									&mut *a_ptr.add(j + l),
									omega,
									omega_shoup,
								);
							}
						}
					}
					4 => {
						for i in 0..(m / 2) {
							let o1 = *self.omegas.get_unchecked(k);
							let o2 = *self.omegas.get_unchecked(k + 1);
							let os1 = *self.omegas_shoup.get_unchecked(k);
							let os2 = *self.omegas_shoup.get_unchecked(k + 1);
							k += 2;

							let omega = Simd::from_array([o1, o1, o1, o1, o2, o2, o2, o2]);
							let omega_shoup =
								Simd::from_array([os1, os1, os1, os1, os2, os2, os2, os2]);

							let s = 2 * i * 8;

							let a = u64x8::from_slice(std::slice::from_raw_parts(a_ptr.add(s), 8));
							let b =
								u64x8::from_slice(std::slice::from_raw_parts(a_ptr.add(s + 8), 8));

							// shuffle
							let x = simd_swizzle!(
								a,
								b,
								[
									First(0),
									First(1),
									First(2),
									First(3),
									Second(0),
									Second(1),
									Second(2),
									Second(3)
								]
							);
							let y = simd_swizzle!(
								a,
								b,
								[
									First(4),
									First(5),
									First(6),
									First(7),
									Second(4),
									Second(5),
									Second(6),
									Second(7)
								]
							);

							let (x, y) = self.butterfly_simd::<8>(x, y, omega, omega_shoup);

							// shuffle back
							let xr = x.as_array();
							let yr = y.as_array();

							let view = std::slice::from_raw_parts_mut(a_ptr.add(s), 16);
							view[..4].copy_from_slice(&xr[..4]);
							view[4..8].copy_from_slice(&yr[..4]);
							view[8..12].copy_from_slice(&xr[4..]);
							view[12..].copy_from_slice(&yr[4..]);
						}
					}

					_ => {
						for i in 0..m {
							let omega = Simd::splat(*self.omegas.get_unchecked(k));
							let omega_shoup = Simd::splat(*self.omegas_shoup.get_unchecked(k));
							k += 1;

							let s = 2 * i * l;

							let (x, _) =
								std::slice::from_raw_parts_mut(a_ptr.add(s), l).as_chunks_mut();
							let (y, _) =
								std::slice::from_raw_parts_mut(a_ptr.add(s + l), l).as_chunks_mut();
							izip!(x, y).for_each(|(_x, _y)| {
								let (xr, yr) = self.butterfly_simd::<8>(
									Simd::from_slice(_x),
									Simd::from_slice(_y),
									omega,
									omega_shoup,
								);
								*_x = *xr.as_array();
								*_y = *yr.as_array();
							});
						}
					}
				}
			}
			l >>= 1;
			m <<= 1;
		}

		// reduce x in [0, 4p) to [0, p)
		let (x, _) = a.as_chunks_mut();
		let p_twice = Simd::splat(self.p_twice);
		let p = Simd::splat(self.p.p);
		x.iter_mut().for_each(|v: &mut [u64; 8]| {
			let mut _x = Simd::from_slice(v);
			_x = _x.simd_min(_x - p_twice);
			_x = _x.simd_min(_x - p);
			*v = *_x.as_array();
		});
	}

	pub fn backward_simd(&self, a: &mut [u64]) {
		let n = self.size;
		let a_ptr = a.as_mut_ptr();

		let mut l = 1;
		let mut m = n >> 1;
		let mut k = 0;

		while m > 0 {
			unsafe {
				lane_unroll!(
					self,
					inv_butterfly_simd,
					zetas_inv,
					zetas_inv_shoup,
					l,
					m,
					k,
					a_ptr
				);
			}

			l <<= 1;
			m >>= 1;
		}

		let size_inv = Simd::splat(self.size_inv);
		let size_inv_shoup = Simd::splat(self.size_inv_shoup);

		let (x, x1) = a.as_chunks_mut();
		debug_assert!(x1.is_empty());
		x.iter_mut().for_each(|v| {
			let mut _x = Simd::from_array(*v);
			_x = self
				.p
				.lazy_mul_shoup_simd::<8>(_x, size_inv, size_inv_shoup);
			_x = _x.simd_min(_x - Simd::splat(self.p.p));
			*v = *_x.as_array();
		});
	}

	pub unsafe fn forward_lazy_simd(&self, a: &mut [u64]) {
		// debug_assert!(LANES == 8);

		let n = self.size;
		let a_ptr = a.as_mut_ptr();

		let mut l = n >> 1;
		let mut m = 1;
		let mut k = 1;

		while l > 0 {
			unsafe {
				lane_unroll!(self, butterfly_simd, omegas, omegas_shoup, l, m, k, a_ptr);
			}
			l >>= 1;
			m <<= 1;
		}
	}

	#[inline]
	fn butterfly_simd<const LANES: usize>(
		&self,
		mut x: Simd<u64, LANES>,
		mut y: Simd<u64, LANES>,
		w: Simd<u64, LANES>,
		w_shoup: Simd<u64, LANES>,
	) -> (Simd<u64, LANES>, Simd<u64, LANES>)
	where
		LaneCount<LANES>: SupportedLaneCount,
	{
		// reduce x to [0, 2p)
		let p_twice = Simd::splat(self.p_twice);
		x = x.simd_min(x - p_twice);

		let t = self.p.lazy_mul_shoup_simd(y, w, w_shoup);
		y = x + p_twice - t;
		x += t;

		(x, y)
	}

	#[inline]
	fn inv_butterfly_simd<const LANES: usize>(
		&self,
		mut x: Simd<u64, LANES>,
		mut y: Simd<u64, LANES>,
		z: Simd<u64, LANES>,
		z_shoup: Simd<u64, LANES>,
	) -> (Simd<u64, LANES>, Simd<u64, LANES>)
	where
		LaneCount<LANES>: SupportedLaneCount,
	{
		let p_twice = Simd::splat(self.p_twice);
		//
		// TODO replace this reduce with reduce in hexl implementation
		let t = x;
		x += y;
		x = x.simd_min(x - p_twice);

		y = self.p.lazy_mul_shoup_simd((p_twice + t - y), z, z_shoup);

		(x, y)
	}

	pub fn forward_hexl(&self, a: &mut [u64]) {
		self.ntt_hexl.forward(a, 1, 1);
	}

	pub fn backward_hexl(&self, a: &mut [u64]) {
		self.ntt_hexl.backward(a, 1, 1);
	}
}

#[cfg(test)]
mod tests {
	use rand::thread_rng;

	use super::{supports_ntt, NttOperator};
	use crate::zq::Modulus;

	#[test]
	fn constructor() {
		for size in [8, 1024] {
			for p in [1153, 4611686018326724609] {
				let q = Modulus::new(p).unwrap();
				let supports_ntt = supports_ntt(p, size);

				let op = NttOperator::new(&q, size);

				if supports_ntt {
					assert!(op.is_some());
				} else {
					assert!(op.is_none());
				}
			}
		}
	}

	#[test]
	fn bijection() {
		let ntests = 100;
		let mut rng = thread_rng();

		for size in [8, 1024] {
			for p in [1153, 4611686018326724609] {
				let q = Modulus::new(p).unwrap();

				if supports_ntt(p, size) {
					let op = NttOperator::new(&q, size).unwrap();

					for _ in 0..ntests {
						let mut a = q.random_vec(size, &mut rng);
						let a_clone = a.clone();
						let mut b = a.clone();

						op.forward(&mut a);
						assert_ne!(a, a_clone);

						unsafe { op.forward_vt(b.as_mut_ptr()) }
						assert_eq!(a, b);

						op.backward(&mut a);
						assert_eq!(a, a_clone);

						unsafe { op.backward_vt(b.as_mut_ptr()) }
						assert_eq!(a, b);
					}
				}
			}
		}
	}

	#[test]
	fn forward_lazy() {
		let ntests = 100;
		let mut rng = thread_rng();

		for size in [8, 1024] {
			for p in [1153, 4611686018326724609] {
				let q = Modulus::new(p).unwrap();

				if supports_ntt(p, size) {
					let op = NttOperator::new(&q, size).unwrap();

					for _ in 0..ntests {
						let mut a = q.random_vec(size, &mut rng);
						let mut a_lazy = a.clone();

						op.forward(&mut a);

						unsafe {
							op.forward_vt_lazy(a_lazy.as_mut_ptr());
							q.reduce_vec(&mut a_lazy);
						}

						assert_eq!(a, a_lazy);
					}
				}
			}
		}
	}

	#[test]
	fn forward_simd_works() {
		let mut rng = thread_rng();
		for size in [32, 1024] {
			for p in [1153, 4611686018326724609] {
				let q = Modulus::new(p).unwrap();

				if supports_ntt(p, size) {
					let op = NttOperator::new(&q, size).unwrap();

					for _ in 0..100 {
						let mut a = q.random_vec(size, &mut rng);
						let mut a_lazy = a.clone();
						let mut b = a.clone();
						let mut b_lazy = a.clone();

						op.forward(&mut a);
						op.forward_simd(&mut b);
						assert_eq!(a, b);

						unsafe {
							op.forward_vt_lazy(a_lazy.as_mut_ptr());
							op.forward_lazy_simd(&mut b_lazy);
							assert_eq!(a_lazy, b_lazy);
						}
					}
				}
			}
		}
	}

	#[test]
	fn backward_simd_works() {
		let mut rng = thread_rng();
		for size in [32, 1024] {
			for p in [1153, 4611686018326724609] {
				if supports_ntt(p, size) {
					let q = Modulus::new(p).unwrap();

					let op = NttOperator::new(&q, size).unwrap();

					for _ in 0..100 {
						let mut a = q.random_vec(size, &mut rng);
						let mut a_clone = a.clone();
						op.forward_simd(&mut a);
						assert_ne!(a_clone, a);

						op.backward_simd(&mut a);
						assert_eq!(a_clone, a);
					}
				}
			}
		}
	}

	#[test]
	fn error_test() {
		let mut rng = thread_rng();
		for size in [32, 1024] {
			for p in [1153] {
				if supports_ntt(p, size) {
					let q = Modulus::new(p).unwrap();

					let op = NttOperator::new(&q, size).unwrap();

					for _ in 0..100 {
						let mut a = q.random_vec(size, &mut rng);
						let mut a_clone = a.clone();

						op.forward_simd(&mut a);

						op.backward_simd(&mut a);
						assert_eq!(a_clone, a);

						// let mut a_clone2 = a.clone();

						// op.backward(&mut a_clone);
						// op.backward_simd::<8>(&mut a_clone2);
						// assert_eq!(a_clone, a_clone2);
					}
				}
			}
		}
	}
}
